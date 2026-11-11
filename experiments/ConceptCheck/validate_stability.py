"""Short reproducibility and representation-health checks before a long run.

This deliberately launches only a shared warmup, short replicated continuations
and one bounded stress continuation. It never launches the 15M experiment.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from shutil import copyfile
import struct
import subprocess
import sys

import numpy as np
import torch

if __package__:
    from .checkpoints import load_training_checkpoint
    from .health import inspect_network_health
    from .neural_network import BreakoutBasicNetwork, configure_activation
    from .process_control import terminate_process_tree
else:
    from checkpoints import load_training_checkpoint
    from health import inspect_network_health
    from neural_network import BreakoutBasicNetwork, configure_activation
    from process_control import terminate_process_tree


ROOT = Path(__file__).resolve().parents[2]
LEARNING_COMPONENTS = (
    "policy_net", "target_net", "optimizer", "replay_buffer", "state", "rng", "environment",
    "health_state",
)
COUNTERS = (
    "step", "epsilon", "episode_reward", "last_episode_reward", "episodes_completed",
    "recent_loss", "policy_training", "target_training",
)
RELATIVE_VARIATION_FLOOR = 0.01
FEATURE_STD_FLOOR = 1e-6
Q_STD_FLOOR = 1e-6
CONV_GRADIENT_NORM_FLOOR = 1e-8
CALIBRATION_SCOPE = (
    "Conservative short-validation engineering gate, not a scientific guarantee of "
    "representation quality or future game scores. Fixed-image feature/Q variation "
    "must retain at least 1% of its warmup value and exceed absolute numeric floors."
)


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash_value(digest, value):
    """Hash values, not pickle object identities, without copying a large replay."""
    if value is None:
        digest.update(b"none;")
    elif isinstance(value, bool):
        digest.update(b"true;" if value else b"false;")
    elif isinstance(value, int):
        digest.update(b"int:" + str(value).encode() + b";")
    elif isinstance(value, float):
        digest.update(b"float:" + struct.pack("!d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"str:" + str(len(encoded)).encode() + b":" + encoded)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"bytes:" + str(len(value)).encode() + b":")
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch:" + str(tensor.dtype).encode())
        _hash_value(digest, list(tensor.shape))
        # Checkpoints here use the ordinary numeric dtypes supported by NumPy.
        digest.update(memoryview(tensor.numpy()).cast("B"))
    elif isinstance(value, np.ndarray):
        digest.update(b"numpy:" + value.dtype.str.encode())
        _hash_value(digest, list(value.shape))
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    elif isinstance(value, np.generic):
        _hash_value(digest, value.item())
    elif isinstance(value, np.random.Generator):
        digest.update(b"generator:")
        _hash_value(digest, value.bit_generator.state)
    elif isinstance(value, dict):
        digest.update(b"dict:" + str(len(value)).encode() + b":")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list, deque)):
        digest.update(type(value).__name__.encode() + b":" + str(len(value)).encode() + b":")
        if isinstance(value, deque):
            _hash_value(digest, value.maxlen)
        for item in value:
            _hash_value(digest, item)
    else:
        raise TypeError(f"Cannot canonically hash checkpoint value {type(value)!r}")


def digest_value(value):
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def checkpoint_fingerprints(path, expected_step):
    checkpoint = load_training_checkpoint(path)
    try:
        if checkpoint["step"] != expected_step:
            raise ValueError(f"{path}: expected step {expected_step}, got {checkpoint['step']}")
        config = checkpoint["config"]
        if config.get("activation") != "leaky-relu" or config.get("deterministic") is not True:
            raise ValueError("Validation requires leaky-relu and deterministic training")
        if checkpoint["environment"]["kind"] != "ale":
            raise ValueError("Validation requires exact ALE state capture")
        fingerprints = {key: digest_value(checkpoint[key]) for key in LEARNING_COMPONENTS}
        fingerprints["counters"] = digest_value({key: checkpoint.get(key) for key in COUNTERS})
        return {
            "step": checkpoint["step"], "replay_size": len(checkpoint["replay_buffer"]),
            "fingerprints": fingerprints,
        }
    finally:
        del checkpoint
        gc.collect()


def fixed_probe_from_checkpoint(path, count=16):
    """Use real, evenly spaced images without sampling or advancing any RNG."""
    checkpoint = load_training_checkpoint(path)
    try:
        replay = checkpoint["replay_buffer"]
        if len(replay) < count:
            raise ValueError("Not enough replay observations for the fixed image probe")
        indices = np.linspace(0, len(replay) - 1, count, dtype=np.int64)
        observations = np.stack([replay[int(index)][0] for index in indices])
        if not np.any(observations[1:] != observations[:1]):
            raise ValueError("The fixed image probe contains no diverse observations")
        return observations
    finally:
        del checkpoint
        gc.collect()


def variation_thresholds(baseline=None):
    thresholds = {"feature_std_across_states": FEATURE_STD_FLOOR,
                  "q_std_across_states": Q_STD_FLOOR,
                  "conv_gradient_norm": CONV_GRADIENT_NORM_FLOOR}
    if baseline is not None:
        for name in ("feature_std_across_states", "q_std_across_states"):
            value = baseline.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Warmup has no finite positive {name}")
            thresholds[name] = max(thresholds[name], RELATIVE_VARIATION_FLOOR * value)
    return thresholds


def _at_least(value, floor):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= floor


def inspect_fixed_probe(weights_path, observations, thresholds=None):
    """Measure the same saved images at each milestone, separately from training."""
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = BreakoutBasicNetwork(in_channels=observations.shape[1],
                                     n_actions=weights["fc.2.weight"].shape[0])
        configure_activation(model, "leaky-relu")
    model.load_state_dict(weights)
    model.eval()
    batch = torch.from_numpy(observations)
    measurement = inspect_network_health(model, batch)
    with torch.no_grad():
        features = model.conv(batch.float() / 255.0)
        q_values = model.fc(features)
    thresholds = variation_thresholds() if thresholds is None else thresholds
    checks = {
        "supported": measurement.get("supported") is True,
        "diverse_observations": measurement.get("diverse_observations") is True,
        "finite": measurement.get("finite") is True,
        "features_not_all_zero": measurement.get("features_all_zero") is False,
        "features_depend_on_observation": (
            isinstance(measurement.get("feature_std_across_states"), (int, float))
            and math.isfinite(measurement["feature_std_across_states"])
            and measurement["feature_std_across_states"] > 0
        ),
        "q_values_depend_on_observation": (
            isinstance(measurement.get("q_std_across_states"), (int, float))
            and math.isfinite(measurement["q_std_across_states"])
            and measurement["q_std_across_states"] > 0
        ),
        "feature_variation_above_floor": _at_least(
            measurement.get("feature_std_across_states"), thresholds["feature_std_across_states"]),
        "q_variation_above_floor": _at_least(
            measurement.get("q_std_across_states"), thresholds["q_std_across_states"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "measurement": measurement,
            "thresholds": thresholds,
            "features_digest": digest_value(features), "q_values_digest": digest_value(q_values)}


def read_health(weights_path, expected_step):
    health_path = weights_path.with_suffix(".health.json")
    health = json.loads(health_path.read_text(encoding="utf-8"))
    checks = {
        "at_expected_step": health.get("step") == expected_step,
        "supported": health.get("supported") is True,
        "diverse_observations": health.get("diverse_observations") is True,
        "finite": health.get("finite") is True,
        "features_not_all_zero": health.get("features_all_zero") is False,
        "features_depend_on_observation": (
            isinstance(health.get("feature_std_across_states"), (int, float))
            and math.isfinite(health["feature_std_across_states"])
            and health["feature_std_across_states"] > 0
        ),
        "conv_gradients_present": health.get("conv_gradients_present") is True,
        "conv_gradients_nonzero": (
            isinstance(health.get("conv_gradient_norm"), (int, float))
            and math.isfinite(health["conv_gradient_norm"])
            and health["conv_gradient_norm"] > 0
        ),
        "feature_variation_above_numeric_floor": _at_least(
            health.get("feature_std_across_states"), FEATURE_STD_FLOOR),
        "q_variation_above_numeric_floor": _at_least(
            health.get("q_std_across_states"), Q_STD_FLOOR),
        "conv_gradient_norm_above_numeric_floor": _at_least(
            health.get("conv_gradient_norm"), CONV_GRADIENT_NORM_FLOOR),
    }
    return {"path": str(health_path), "passed": all(checks.values()),
            "checks": checks, "measurement": health}


def audit_validation_report(run_dir):
    """Apply calibrated gates to an already finished run, including old wrappers.

    This reads saved artifacts on CPU and only adds the baseline copy/report. It
    never starts or resumes training and refuses to audit a running wrapper.
    """
    directory = Path(run_dir).resolve()
    report_path = directory / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") not in ("completed", "failed"):
        raise ValueError("Wait for validation to finish before auditing its artifacts")
    original_passed = report.setdefault("pre_calibration_passed", report.get("passed") is True)
    calibrated = {"passed": False, "scope": CALIBRATION_SCOPE, "relative_floor": RELATIVE_VARIATION_FLOOR,
                  "numeric_floors": variation_thresholds(), "stages": {}, "audited_at_utc": _timestamp()}
    try:
        probe_path = Path(report.get("probe_observations", {}).get(
            "path", directory / "probe-observations.npy"))
        probe = np.load(probe_path, allow_pickle=False)
        expected_digest = report.get("probe_observations", {}).get("digest")
        if expected_digest is not None and digest_value(probe) != expected_digest:
            raise ValueError("Fixed probe observations no longer match the recorded digest")
        baseline_weights = directory / "probe-baseline.pth"
        if not baseline_weights.exists():
            warmup_weights = Path(report["stages"]["warmup"]["weights"])
            copyfile(warmup_weights, baseline_weights)
        baseline = inspect_fixed_probe(baseline_weights, probe)
        thresholds = variation_thresholds(baseline["measurement"])
        calibrated.update(thresholds=thresholds, baseline={"weights": str(baseline_weights),
                          "observations": str(probe_path), "observation_digest": digest_value(probe),
                          "inspection": baseline})
        fingerprints = {}
        for name in ("replica-a", "replica-b", "paced-comparison"):
            info = report["stages"].get(name)
            if info is not None:
                fingerprints[name] = checkpoint_fingerprints(Path(info["checkpoint"]), info["stop_at"])
        if "replica-a" not in fingerprints or "replica-b" not in fingerprints:
            raise ValueError("Both replica checkpoints are required for calibrated audit")
        first, second = fingerprints["replica-a"], fingerprints["replica-b"]
        report["repeatability"] = {
            "passed": first == second, "replicas": {name: fingerprints[name] for name in ("replica-a", "replica-b")},
            "different_components": [key for key, value in first["fingerprints"].items()
                                     if value != second["fingerprints"][key]],
        }
        if "paced-comparison" in fingerprints:
            report["pacing_invariance"] = {
                "passed": fingerprints["paced-comparison"] == first,
                "comparison_rate": report.get("config", {}).get("comparison_rate"),
                "checkpoint": fingerprints["paced-comparison"],
            }
        calibrated["exact_comparison_scope"] = {
            "components": list(LEARNING_COMPONENTS), "counters": list(COUNTERS),
            "excluded": ["elapsed_seconds", "config (output paths and pacing differ intentionally)"],
        }
        names = ["replica-a", "replica-b", "stress"]
        if "paced-comparison" in report["stages"]:
            names.insert(2, "paced-comparison")
        for name in names:
            info = report["stages"].get(name)
            if info is None:
                calibrated["stages"][name] = {"passed": False, "error": "Stage did not complete"}
                continue
            weights = Path(info["weights"])
            fixed = inspect_fixed_probe(weights, probe, thresholds)
            health = read_health(weights, info["stop_at"])
            calibrated["stages"][name] = {"passed": fixed["passed"] and health["passed"],
                                          "fixed_probe": fixed, "training_health": health}
        calibrated["passed"] = (baseline["passed"] and report["repeatability"]["passed"]
                                and report.get("pacing_invariance", {"passed": True})["passed"]
                                and all(stage["passed"] for stage in calibrated["stages"].values()))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        calibrated["error"] = repr(error)
    report["calibrated_health"] = calibrated
    report["passed"] = original_passed and calibrated["passed"]
    report["updated_at_utc"] = _timestamp()
    if not calibrated["passed"] and report.get("status") == "completed":
        report["status"] = "failed"
        report["error"] = "Calibrated representation-health gate failed"
    _write_json(report_path, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=230849415)
    parser.add_argument("--warmup-step", type=int, default=19_999)
    parser.add_argument("--end-step", type=int, default=30_000)
    parser.add_argument("--stress-steps", type=int, default=100_000)
    parser.add_argument("--max-steps-per-second", type=float, default=220.0)
    parser.add_argument("--comparison-rate", type=float, default=None,
                        help="Optional third continuation at a different pacing rate")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in uint32 range")
    if not 0 < args.warmup_step < 20_000 < args.end_step < args.stress_steps:
        parser.error("require 0 < warmup-step < 20000 < end-step < stress-steps")
    for name in ("max_steps_per_second", "comparison_rate"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    args.run_dir = str(Path(args.run_dir).resolve())
    if (Path(args.run_dir) / "validation.json").exists():
        parser.error("validation output already exists; use a fresh run directory")
    return args


def run_validation(argv=None):
    args = parse_args(argv)
    directory = Path(args.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "validation.json"
    report = {
        "status": "starting", "passed": False, "created_at_utc": _timestamp(),
        "config": vars(args), "stages": {}, "repeatability": {}, "health": {}, "fixed_probe": {},
        "scope": "Short controlled validation, not proof of long-run stability or game skill",
        "error": None,
    }

    def update(**changes):
        report.update(changes)
        report["updated_at_utc"] = _timestamp()
        _write_json(report_path, report)

    def train_stage(name, stop_at, *, resume=None, rate=None):
        stage = directory / name
        stage.mkdir(parents=True, exist_ok=True)
        weights = stage / "model.pth"
        checkpoint = weights.with_suffix(".training.pt")
        command = [sys.executable, "-u", "-m", "experiments.ConceptCheck.train",
                   "--device", args.device, "--steps", str(args.stress_steps),
                   "--stop-at", str(stop_at), "--save-path", str(weights),
                   "--checkpoint-path", str(checkpoint),
                   "--max-steps-per-second", str(args.max_steps_per_second if rate is None else rate)]
        if resume is None:
            command += ["--algorithm", "double-dqn", "--seed", str(args.seed),
                        "--activation", "leaky-relu", "--deterministic"]
        else:
            command += ["--resume", str(resume)]
        info = {"command": command, "started_at_utc": _timestamp(), "stop_at": stop_at,
                "checkpoint": str(checkpoint), "weights": str(weights)}
        report["stages"][name] = info
        update(status="running", current_stage=name)
        print(f"Validation stage {name}: decisions through {stop_at:,}", flush=True)
        with (stage / "stdout.log").open("a", encoding="utf-8") as stdout, \
                (stage / "stderr.log").open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            info["pid"] = process.pid
            update()
            try:
                returncode = process.wait()
            except BaseException:
                # Windows virtualenv launchers may own a second Python process.
                try:
                    info["child_cleanup"] = terminate_process_tree(process, timeout=15)
                except Exception as cleanup_error:
                    info.update(cleanup_error=repr(cleanup_error),
                                child_active_on_cleanup_error=process.poll() is None)
                    update()
                    raise
                info.update(returncode=process.returncode, interrupted_at_utc=_timestamp())
                raise
        info.update(returncode=returncode, completed_at_utc=_timestamp())
        update()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        progress = json.loads(weights.with_suffix(".progress.json").read_text(encoding="utf-8"))
        if progress["step"] != stop_at or progress["status"] not in ("paused", "completed"):
            raise ValueError(f"{name} did not stop cleanly at {stop_at}")
        return weights, checkpoint

    try:
        warmup_weights, warmup = train_stage("warmup", args.warmup_step)
        probe = fixed_probe_from_checkpoint(warmup)
        probe_path = directory / "probe-observations.npy"
        np.save(probe_path, probe)
        report["probe_observations"] = {"path": str(probe_path), "shape": list(probe.shape),
                                        "digest": digest_value(probe), "source": str(warmup)}
        baseline_weights = directory / "probe-baseline.pth"
        copyfile(warmup_weights, baseline_weights)
        baseline = inspect_fixed_probe(baseline_weights, probe)
        thresholds = variation_thresholds(baseline["measurement"])
        report["calibration"] = {"scope": CALIBRATION_SCOPE, "relative_floor": RELATIVE_VARIATION_FLOOR,
                                  "numeric_floors": variation_thresholds(), "thresholds": thresholds,
                                  "baseline_weights": str(baseline_weights), "baseline": baseline}
        update()
        if not baseline["passed"]:
            raise RuntimeError("Warmup representation has insufficient fixed-probe variation")
        fingerprints = {}
        for name, rate in (("replica-a", args.max_steps_per_second),
                           ("replica-b", args.max_steps_per_second)):
            weights, checkpoint = train_stage(name, args.end_step, resume=warmup, rate=rate)
            fingerprints[name] = checkpoint_fingerprints(checkpoint, args.end_step)
            report["health"][name] = read_health(weights, args.end_step)
            report["fixed_probe"][name] = inspect_fixed_probe(weights, probe, thresholds)
            update()
        report["repeatability"] = {
            "passed": fingerprints["replica-a"] == fingerprints["replica-b"],
            "different_components": [key for key, value in fingerprints["replica-a"]["fingerprints"].items()
                                     if value != fingerprints["replica-b"]["fingerprints"][key]],
            "replicas": fingerprints,
        }
        update()
        if not report["repeatability"]["passed"]:
            raise RuntimeError("Replicated continuations differ; stress training was not started")
        if (not all(value["passed"] for value in report["health"].values())
                or not all(value["passed"] for value in report["fixed_probe"].values())):
            raise RuntimeError("Replica representation health failed; stress training was not started")
        if args.comparison_rate is not None:
            weights, checkpoint = train_stage("paced-comparison", args.end_step,
                                              resume=warmup, rate=args.comparison_rate)
            comparison = checkpoint_fingerprints(checkpoint, args.end_step)
            report["pacing_invariance"] = {
                "passed": comparison == fingerprints["replica-a"],
                "comparison_rate": args.comparison_rate, "checkpoint": comparison,
            }
            report["health"]["paced-comparison"] = read_health(weights, args.end_step)
            report["fixed_probe"]["paced-comparison"] = inspect_fixed_probe(weights, probe, thresholds)
            update()
            if (not report["pacing_invariance"]["passed"]
                    or not report["health"]["paced-comparison"]["passed"]
                    or not report["fixed_probe"]["paced-comparison"]["passed"]):
                raise RuntimeError("Pacing comparison failed; stress training was not started")
        # Keep A's 30k files intact; the stress continuation has its own output.
        stress_weights, stress_checkpoint = train_stage(
            "stress", args.stress_steps,
            resume=directory / "replica-a" / "model.training.pt",
        )
        report["stress"] = checkpoint_fingerprints(stress_checkpoint, args.stress_steps)
        report["health"]["stress"] = read_health(stress_weights, args.stress_steps)
        report["fixed_probe"]["stress"] = inspect_fixed_probe(stress_weights, probe, thresholds)
        passed = (report["repeatability"]["passed"]
                  and all(value["passed"] for value in report["health"].values())
                  and all(value["passed"] for value in report["fixed_probe"].values())
                  and report.get("pacing_invariance", {"passed": True})["passed"])
        update(status="completed", passed=passed, current_stage=None,
               completed_at_utc=_timestamp())
        audited = audit_validation_report(directory)
        report.clear()
        report.update(audited)
        passed = report["passed"]
        if not passed:
            raise RuntimeError("Short validation failed; inspect component fingerprints and health checks")
    except BaseException as error:
        update(status="failed", passed=False, error=repr(error))
        raise
    return report


if __name__ == "__main__":
    run_validation()
