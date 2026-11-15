"""Evaluate one policy in unmodified Breakout on a shared set of game seeds."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

if __package__:
    from .model import Network, configure
    from .rewards import make_env
else:
    from model import Network, configure
    from rewards import make_env


def evaluate(model_path, output, episodes=100, seed=1847319063, device="cuda", max_steps=27000):
    configure(device)
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    model = Network().to(device)
    model.load_state_dict(saved["policy_net"])
    model.eval()
    cfg = saved.get("config", {})
    env = make_env("raw", cfg.get("mode", 0), cfg.get("difficulty", 0), cfg.get("frame_skip", 4),
                   cfg.get("sticky_action_probability", 0.25), cfg.get("gamma", 0.995))
    records = []
    started = time.perf_counter()
    try:
        for episode in range(episodes):
            game_seed = (seed + episode) % 2**32
            env.action_space.seed(game_seed)
            state, info = env.reset(seed=game_seed)
            previous_lives, fire, score = info["lives"], True, 0.0
            for step in range(1, max_steps + 1):
                with torch.inference_mode():
                    observation = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(device)
                    action = 1 if fire else int(model(observation).argmax(1).item())
                state, reward, terminated, truncated, info = env.step(action)
                score += float(reward)
                fire = info["lives"] < previous_lives
                previous_lives = info["lives"]
                if terminated or truncated:
                    break
            records.append({"seed": game_seed, "reward": score, "steps": step,
                            "capped": bool(truncated or not terminated)})
    finally:
        env.close()
    scores = [record["reward"] for record in records]
    with Path(model_path).open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "checkpoint_sha256": checkpoint_sha256, "training_step": saved.get("step"),
        "evaluation_seed": seed, "max_steps": max_steps,
        "episode_seeds": [record["seed"] for record in records], "episodes": records,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": {"episodes": len(scores), "mean_reward": statistics.mean(scores),
                    "median_reward": statistics.median(scores), "max_reward": max(scores),
                    "std_reward": statistics.pstdev(scores),
                    "wins_864": sum(score >= 864 for score in scores),
                    "hit_step_limit": sum(record["capped"] for record in records)},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1847319063)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=27000)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or not 0 <= args.seed < 2**32:
        parser.error("episodes/max-steps must be positive and seed must fit uint32")
    print(json.dumps(evaluate(args.model, args.output, args.episodes, args.seed,
                             args.device, args.max_steps)["summary"]))
