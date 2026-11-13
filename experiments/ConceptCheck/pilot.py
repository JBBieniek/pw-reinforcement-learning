"""Compare four conservative Double DQN recipes with one GPU child at a time."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import zipfile

if __package__:
    from . import experiment
else:
    import experiment


VARIANTS = (
    ("control", {}),
    ("life_loss", {"terminal_on_life_loss": True}),
    ("replay200k", {"buffer_capacity": 200_000}),
    ("slow_exploration", {"epsilon_decay_steps": 1_000_000}),
)
DEFAULTS = {
    "milestones": [1_000_000, 2_000_000], "seed": 230849415,
    "eval_seed": 3303497698, "eval_episodes": 100, "max_steps": 27_000,
    "device": "cuda", "max_steps_per_second": 0.0,
}


def _source_files():
    paths = [experiment.ROOT / "README.md", experiment.ROOT / "requirements.txt"]
    for folder in ("experiments", "tests"):
        paths.extend((experiment.ROOT / folder).rglob("*.py"))
    return sorted(path for path in paths if path.is_file())


def _source_manifest():
    return {path.relative_to(experiment.ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in _source_files()}


def _archive_source(directory, manifest):
    destination = directory / "source.zip"
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, digest in manifest.items():
                data = (experiment.ROOT / relative).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise RuntimeError("source changed while creating the pilot archive; retry")
                archive.writestr(relative, data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _safe_source_path(root, relative):
    if not isinstance(relative, str):
        raise ValueError("unsafe path in pilot source manifest")
    path = PurePosixPath(relative)
    if (path.is_absolute() or not path.parts
            or path.as_posix() != relative or "\\" in relative
            or any(part in (".", "..") or ":" in part for part in path.parts)):
        raise ValueError("unsafe path in pilot source manifest")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("pilot source path escapes its frozen directory")
    return destination


def _verify_frozen_source(root, manifest):
    for relative, digest in manifest.items():
        path = _safe_source_path(root, relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen pilot source is missing or has changed: {relative}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(manifest):
        raise ValueError("frozen pilot source contains unexpected files")


def _frozen_source(directory, config):
    """Run every phase from verified archive bytes, never the live working tree."""
    archive_path = directory / "source.zip"
    if (not archive_path.is_file()
            or hashlib.sha256(archive_path.read_bytes()).hexdigest() != config["source_zip_sha256"]):
        raise ValueError("pilot source archive is missing or has changed")
    manifest = config["source_files_sha256"]
    destination = directory / "source"
    for relative in manifest:
        _safe_source_path(destination, relative)
    if destination.exists():
        _verify_frozen_source(destination, manifest)
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".source-", dir=directory))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(manifest):
                raise ValueError("pilot source archive differs from its manifest")
            for relative, digest in manifest.items():
                target = _safe_source_path(temporary, relative)
                data = archive.read(relative)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"pilot source archive member differs: {relative}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        _verify_frozen_source(temporary, manifest)
        # Both paths are direct children of the explicitly selected run directory.
        if temporary.resolve().parent != directory.resolve() or destination.resolve().parent != directory.resolve():
            raise ValueError("frozen source targets escape the pilot directory")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            if temporary.resolve().parent != directory.resolve():
                raise ValueError("temporary source directory escaped the pilot directory")
            shutil.rmtree(temporary)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--milestones", type=experiment._milestones)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-steps-per-second", type=float)
    args = parser.parse_args(argv)
    args.run_dir = str(Path(args.run_dir).resolve())
    path = Path(args.run_dir) / "pilot.config.json"
    saved = None
    if args.resume:
        if not path.is_file():
            parser.error("--resume requires an existing pilot.config.json")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("format_version") != 1 or saved.get("variants") != _variants():
            parser.error("unsupported or changed pilot plan")
    elif path.exists():
        parser.error("pilot already exists; use --resume")
    for name, default in DEFAULTS.items():
        value = getattr(args, name)
        expected = saved[name] if saved else default
        if saved and value is not None and value != expected and name not in ("device", "max_steps_per_second"):
            parser.error("--" + name.replace("_", "-") + " differs from the saved pilot")
        if value is None:
            setattr(args, name, expected)
    if any(not 0 <= seed < 2**32 for seed in (args.seed, args.eval_seed)):
        parser.error("seeds must be between 0 and 4294967295")
    if args.eval_episodes <= 0 or args.max_steps <= 0:
        parser.error("--eval-episodes and --max-steps must be positive")
    if not math.isfinite(args.max_steps_per_second) or args.max_steps_per_second < 0:
        parser.error("--max-steps-per-second must be finite and non-negative")
    args._saved_config = saved
    return args


def _variants():
    base = {**experiment.TRAINING_DEFAULTS, "replay_storage": "frames"}
    return [{"name": name, "training": {**base, **changes}} for name, changes in VARIANTS]


def _experiment_arguments(args, variant, *, resume):
    command = ["--run-dir", str(Path(args.run_dir) / variant["name"]),
               "--algorithm", "double-dqn", "--activation", "leaky-relu", "--deterministic",
               "--milestones", ",".join(map(str, args.milestones)),
               "--seed", str(args.seed), "--eval-seed", str(args.eval_seed),
               "--eval-episodes", str(args.eval_episodes), "--max-steps", str(args.max_steps),
               "--device", args.device, "--max-steps-per-second", str(args.max_steps_per_second)]
    for name, value in variant["training"].items():
        option = "--" + name.replace("_", "-")
        command.extend([option if value else "--no-terminal-on-life-loss"]
                       if name == "terminal_on_life_loss" else [option, str(value)])
    if resume:
        command.append("--resume-experiment")
    return command


def _collect_variant(args, variant):
    directory = Path(args.run_dir) / variant["name"]
    progress = json.loads((directory / "experiment.progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed" or progress.get("completed_milestones") != args.milestones:
        raise ValueError(f"variant {variant['name']} did not complete its requested milestones")
    settings = experiment.parse_args(_experiment_arguments(args, variant, resume=True))
    rows = []
    for milestone in args.milestones:
        report = experiment._evaluation_report(experiment._evaluation_path(directory, milestone),
                                              experiment._weights_path(directory, milestone),
                                              milestone, settings, cached=True)
        if report is None:
            raise ValueError(f"variant {variant['name']} has an invalid evaluation at {milestone}")
        rows.append({"variant": variant["name"], "decisions": milestone,
                     "algorithm": "double-dqn", **report["summary"]})
    return rows


def _write_comparison(directory, rows):
    path = directory / "comparison.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = ["variant", "decisions", "algorithm", *experiment.SUMMARY_FIELDS]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    os.replace(temporary, path)


def run_pilot(argv=None):
    args = parse_args(argv)
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "pilot.config.json"
    progress_path = directory / "pilot.progress.json"
    started = time.monotonic()
    progress = {"status": "starting", "pid": os.getpid(), "child_pid": None,
                "child_cleanup": None, "current_variant": None, "completed_variants": [],
                "results": [], "error": None, "elapsed_seconds": 0.0}

    def update(**changes):
        progress.update(changes)
        progress["elapsed_seconds"] = round(time.monotonic() - started, 3)
        progress["updated_at_utc"] = experiment._timestamp()
        experiment._write_json(progress_path, progress)

    with experiment._experiment_lock(directory):
        if args.resume:
            if json.loads(config_path.read_text(encoding="utf-8")) != args._saved_config:
                raise RuntimeError("pilot configuration changed before its lock was acquired; retry")
            config = args._saved_config.copy()
        else:
            if (config_path.exists() or progress_path.exists()
                    or (directory / "experiment.config.json").exists()
                    or (directory / "model.training.pt").exists()
                    or any((directory / item["name"]).exists() for item in _variants())):
                raise ValueError("pilot directory already contains a run; use --resume")
            manifest = _source_manifest()
            config = {name: getattr(args, name) for name in DEFAULTS}
            config.update(format_version=1, run_dir=args.run_dir, variants=_variants(),
                          created_at_utc=experiment._timestamp(), source_files_sha256=manifest,
                          source_zip_sha256=_archive_source(directory, manifest))
        source_root = _frozen_source(directory, config)
        progress["source_root"] = str(source_root)
        if progress_path.exists():
            previous = json.loads(progress_path.read_text(encoding="utf-8"))
            previous_child = previous.get("child_pid")
            if previous_child is not None and experiment._process_alive(previous_child):
                raise RuntimeError("previous pilot child is still running; wait before resuming")
        config.update(device=args.device, max_steps_per_second=args.max_steps_per_second)
        experiment._write_json(config_path, config)
        progress["started_at_utc"] = config["created_at_utc"]
        update()
        rows = []
        try:
            for variant in config["variants"]:
                nested = directory / variant["name"]
                nested_config = nested / "experiment.config.json"
                update(status="running", current_variant=variant["name"])
                _verify_frozen_source(source_root, config["source_files_sha256"])
                command = [sys.executable, "-B", "-u", "-m", "experiments.ConceptCheck.experiment",
                           *_experiment_arguments(args, variant, resume=nested_config.exists())]
                print(f"Pilot variant: {variant['name']}", flush=True)
                experiment._run_child(command, update, cwd=source_root)
                rows.extend(_collect_variant(args, variant))
                _write_comparison(directory, rows)
                update(completed_variants=[*progress["completed_variants"], variant["name"]],
                       results=rows)
            update(status="completed", current_variant=None)
        except KeyboardInterrupt:
            update(status="interrupted", error="Pilot interrupted; resume from each variant's latest checkpoint")
            raise
        except Exception as error:
            update(status="failed", error=repr(error))
            raise
    return progress


if __name__ == "__main__":
    run_pilot()
