"""Evaluate saved DQN and Double DQN policies on a shared set of episode seeds."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import time

import numpy as np
import torch

if __package__:
    from .neural_network import BreakoutBasicNetwork, configure_activation, make_breakout_env
    from .pacing import StepPacer
    from .runtime import configure_determinism
else:
    from neural_network import BreakoutBasicNetwork, configure_activation, make_breakout_env
    from pacing import StepPacer
    from runtime import configure_determinism


def evaluate_policy(model, env, *, episodes=50, seed=None, device="cpu",
                    max_steps=27_000, auto_fire=True, frame_delay=0.0,
                    max_steps_per_second=0.0):
    """Evaluate without changing model weights; initial FIRE counts toward the cap."""
    seed = secrets.randbits(32) if seed is None else seed
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 4294967295")
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    device = torch.device(device)
    model.eval()
    records = []
    pacer = StepPacer(max_steps_per_second)
    # Identical starting seeds across checkpoints and algorithms; wrap at uint32.
    episode_seeds = [(seed + episode) % 2**32 for episode in range(episodes)]
    for episode_seed in episode_seeds:
        env.action_space.seed(episode_seed)
        state, info = env.reset(seed=episode_seed)
        previous_lives = info.get("lives")
        fire_next = True
        reward_sum = 0.0
        terminated = truncated = False
        for step in range(1, max_steps + 1):
            if fire_next:
                action = 1  # FIRE is action 1 in the minimal Breakout action set.
            else:
                with torch.inference_mode():
                    observation = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(device)
                    action = int(model(observation).argmax(dim=1).item())
            state, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            lives = info.get("lives", previous_lives)
            fire_next = bool(auto_fire and previous_lives is not None
                             and lives is not None and lives < previous_lives)
            previous_lives = lives
            pacer.wait()
            if frame_delay > 0:
                time.sleep(frame_delay)
            if terminated or truncated:
                break
        records.append({
            "seed": int(episode_seed), "reward": reward_sum, "steps": step,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "hit_step_limit": not bool(terminated or truncated),
            "lives_remaining": int(previous_lives) if previous_lives is not None else None,
            "reached_864": reward_sum >= 864,
        })
    rewards = [record["reward"] for record in records]
    wins = sum(record["reached_864"] for record in records)
    return {
        "evaluation_seed": int(seed), "episode_seeds": episode_seeds,
        "policy_epsilon": 0.0, "max_steps": max_steps,
        "auto_fire_after_life_loss": bool(auto_fire),
        "summary": {
            "episodes": len(records), "mean_reward": float(np.mean(rewards)),
            "median_reward": float(np.median(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)), "max_reward": float(np.max(rewards)),
            "wins_864": wins, "win_rate_864": wins / len(records),
            "hit_step_limit": sum(record["hit_step_limit"] for record in records),
            "mean_steps": float(np.mean([record["steps"] for record in records])),
        },
        "episodes": records,
    }


def evaluate(model_path, *, episodes=50, seed=None, device=None, render_mode=None,
             max_steps=27_000, auto_fire=True, mode=0, difficulty=0, frame_skip=4,
             sticky_action_probability=0.25, output_path=None, fps=30.0,
             max_steps_per_second=0.0, activation=None, deterministic=None):
    """Load saved policy weights and return a JSON report without training."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    config_path = model_path.with_suffix(".config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    activation = config.get("activation", "relu") if activation is None else activation
    deterministic = config.get("deterministic", False) if deterministic is None else deterministic
    if activation not in ("relu", "leaky-relu"):
        raise ValueError(f"Unknown activation: {activation!r}")
    runtime_flags = configure_determinism(deterministic)
    torch.set_num_threads(2)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    environment = {
        "mode": mode, "difficulty": difficulty, "frame_skip": frame_skip,
        "sticky_action_probability": sticky_action_probability,
    }
    # Training always saves plain policy weights next to the full resume state.
    # Keep weights_only=True: a policy evaluation must not unpickle arbitrary code.
    try:
        weights = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as error:
        raise ValueError("Cannot load policy weights. For a full training checkpoint, "
                         "evaluate the accompanying policy .pth file instead.") from error
    if "optimizer" in weights or "replay_buffer" in weights:
        raise ValueError("Evaluate the policy .pth file saved alongside the full training checkpoint.")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    env = make_breakout_env(render_mode=render_mode, **environment)
    try:
        model = BreakoutBasicNetwork(in_channels=4, n_actions=env.action_space.n).to(device)
        configure_activation(model, activation)
        model.load_state_dict(weights)
        report = evaluate_policy(
            model, env, episodes=episodes, seed=seed, device=device,
            max_steps=max_steps, auto_fire=auto_fire,
            frame_delay=1.0 / fps if render_mode == "human" and fps > 0 else 0.0,
            max_steps_per_second=max_steps_per_second,
        )
    finally:
        env.close()
    report.update({
        "model_path": str(model_path.resolve()), "device": str(device),
        "environment": environment, "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "max_steps_per_second": max_steps_per_second,
        "activation": activation, "deterministic": deterministic,
        "runtime_flags": runtime_flags,
    })
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained DQN or Double DQN on Breakout")
    parser.add_argument("--algorithm", choices=["dqn", "double-dqn"], default="dqn")
    parser.add_argument("--model-path", help="Saved policy weights (.pth)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Shared evaluation seed (default: generate and save one)")
    parser.add_argument("--max-steps", type=int, default=27_000, help="Decision cap per episode, including FIRE")
    parser.add_argument("--auto-fire", action=argparse.BooleanOptionalAction, default=True,
                        help="Automatically FIRE after a lost life (enabled by default)")
    parser.add_argument("--mode", type=int, default=None)
    parser.add_argument("--difficulty", type=int, choices=[0, 1], default=None)
    parser.add_argument("--frame-skip", type=int, choices=[2, 4, 8], default=None)
    parser.add_argument("--sticky-action-probability", type=float, choices=[0.0, 0.25], default=None)
    parser.add_argument("--render-mode", choices=["human", "none"], default="none")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", help="PyTorch device (default: CUDA when available)")
    parser.add_argument("--activation", choices=["relu", "leaky-relu"], default=None,
                        help="Network activation (default: sidecar config, otherwise legacy ReLU)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None,
                        help="Deterministic numerical operations (default: sidecar config, otherwise off)")
    parser.add_argument("--max-steps-per-second", type=float, default=0.0,
                        help="Soft decision-rate limit (0: unlimited)")
    parser.add_argument("--output", help="JSON report path (default: <model>.evaluation.json)")
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")
    if not np.isfinite(args.max_steps_per_second) or args.max_steps_per_second < 0:
        parser.error("--max-steps-per-second must be finite and non-negative")
    if args.model_path is None:
        args.model_path = str(Path(__file__).with_name(f"breakout_{args.algorithm.replace('-', '_')}.pth"))
    config_path = Path(args.model_path).with_suffix(".config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    for key, fallback in (("mode", 0), ("difficulty", 0), ("frame_skip", 4),
                          ("sticky_action_probability", 0.25), ("activation", "relu"),
                          ("deterministic", False)):
        if getattr(args, key) is None:
            setattr(args, key, config.get(key, fallback))
    if args.output is None:
        args.output = str(Path(args.model_path).with_suffix(".evaluation.json"))
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Evaluating {args.model_path} | Seed: {args.seed} | Episodes: {args.episodes} | "
          f"Activation: {args.activation} | Deterministic: {args.deterministic}", flush=True)
    report = evaluate(
        args.model_path, episodes=args.episodes, seed=args.seed, device=args.device,
        render_mode=None if args.render_mode == "none" else args.render_mode,
        max_steps=args.max_steps, auto_fire=args.auto_fire, mode=args.mode,
        difficulty=args.difficulty, frame_skip=args.frame_skip,
        sticky_action_probability=args.sticky_action_probability,
        output_path=args.output, fps=args.fps,
        max_steps_per_second=args.max_steps_per_second,
        activation=args.activation, deterministic=args.deterministic,
    )
    summary = report["summary"]
    print(f"Mean: {summary['mean_reward']:.2f} | Median: {summary['median_reward']:.2f} | "
          f"Min/Max: {summary['min_reward']:.0f}/{summary['max_reward']:.0f} | "
          f"Std: {summary['std_reward']:.2f}")
    print(f"864: {summary['wins_864']}/{summary['episodes']} "
          f"({summary['win_rate_864']:.1%}) | Step caps: {summary['hit_step_limit']}")
    print(f"Report: {args.output}")
    return report


if __name__ == "__main__":
    main()
