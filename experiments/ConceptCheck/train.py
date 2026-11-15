"""Image DQN/DDQN with uniform replay; checkpoints keep weights and Adam, not replay/game."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import shutil
import time
import numpy as np
import torch
from torch import nn
if __package__:
    from .model import Network, configure
    from .rewards import make_env
else:
    from model import Network, configure
    from rewards import make_env


@torch.no_grad()
def compute_targets(policy, target, next_states, rewards, terminal, gamma, algorithm="double-dqn"):
    next_q = target(next_states)
    if algorithm == "double-dqn":
        next_q = next_q.gather(1, policy(next_states).argmax(1, keepdim=True))
    else:
        next_q = next_q.max(1, keepdim=True).values
    return rewards + gamma * (~terminal).float() * next_q


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                add(key); add(x[key])
        elif isinstance(x, (tuple, list)):
            for item in x: add(item)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def restore_rng(state, device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda": torch.cuda.set_rng_state_all(state["cuda"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, help="Trusted weights/Adam checkpoint; starts a new game and replay")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=500_000)
    parser.add_argument("--reward", choices=["raw", "small-bounce", "landing"], default="raw")
    parser.add_argument("--algorithm", choices=["dqn", "double-dqn"], default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1_000_000, help="Decision interval; 0 disables evaluation")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1847319063)
    for name, kind in (("lr", float), ("gamma", float), ("buffer-capacity", int),
                       ("train-freq", int), ("batch-size", int), ("min-replay-size", int),
                       ("target-update-freq", int), ("mode", int), ("difficulty", int),
                       ("frame-skip", int), ("sticky-action-probability", float)):
        parser.add_argument("--" + name, type=kind, default=None)
    args = parser.parse_args(argv)
    start = torch.load(args.start, map_location="cpu", weights_only=False) if args.start else {}
    previous, cfg = start.get("config", {}), {}
    defaults = dict(lr=5e-5, gamma=0.995, buffer_capacity=500_000, train_freq=2, batch_size=32,
                    min_replay_size=20_000, target_update_freq=10_000, algorithm="double-dqn",
                    mode=0, difficulty=0, frame_skip=4, sticky_action_probability=0.25,
                    epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=1_000_000)
    for name, default in defaults.items():
        value = getattr(args, name, None)
        cfg[name] = value if value is not None else previous.get(name, default)
    cfg.update(reward=args.reward, seed=args.seed if args.seed is not None else previous.get("seed", secrets.randbits(32)))
    if (args.decisions <= 0 or args.eval_every < 0 or args.eval_episodes <= 0 or not 0 <= args.eval_seed < 2**32
            or not 1 <= cfg["batch_size"] <= cfg["min_replay_size"] <= cfg["buffer_capacity"]
            or cfg["train_freq"] <= 0 or not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0
            or cfg["target_update_freq"] <= 0 or cfg["frame_skip"] <= 0
            or not 0 <= cfg["sticky_action_probability"] <= 1
            or not 0 <= cfg["gamma"] <= 1 or not 0 <= cfg["seed"] < 2**32):
        parser.error("Invalid decision budget, replay, learning rate, gamma or seed")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "model.progress.json").exists():
        parser.error("Output directory already contains a run")
    device = torch.device(args.device)
    configure(device)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    env = make_env(args.reward, cfg["mode"], cfg["difficulty"], cfg["frame_skip"],
                   cfg["sticky_action_probability"], cfg["gamma"])
    state, _ = env.reset(seed=cfg["seed"])
    state = np.array(state)
    env.action_space.seed(cfg["seed"])
    policy, target = Network().to(device), Network().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    if start:
        policy.load_state_dict(start["policy_net"])
        target.load_state_dict(start["target_net"])
        optimizer.load_state_dict(start["optimizer"])
        restore_rng(start["rng"], device)
    else:
        target.load_state_dict(policy.state_dict())
    for group in optimizer.param_groups: group["lr"] = cfg["lr"]
    target.eval().requires_grad_(False)
    source_step = int(start.get("step", start.get("source_step", 0)))
    def rng():
        return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
    write_json(args.output / "initial_state_hashes.json", dict(
        policy=digest(policy.state_dict()), target=digest(target.state_dict()),
        optimizer=digest(optimizer.state_dict()), rng=digest(rng()), observation=digest(state)))
    replay, position, updates, trace = [], 0, 0, hashlib.sha256()
    step, episode_score, last_score, aux_total, episode_aux, last_aux = 0, 0., 0., 0., 0., 0.
    loss_value, grad_norm, q_std, feature_std = 0., 0., 0., None
    began, status, error, collapsed_checks = time.perf_counter(), "running", None, 0
    def progress():
        elapsed = time.perf_counter() - began
        write_json(args.output / "model.progress.json", dict(status=status, pid=os.getpid(),
            decisions=step, absolute_step=source_step+step, total_steps=args.decisions,
            training_steps_per_second=step/elapsed if elapsed else None, elapsed_seconds=elapsed,
            epsilon=epsilon, loss=loss_value, gradient_norm=grad_norm, q_std=q_std,
            feature_std=feature_std, replay_size=len(replay), gradient_updates=updates,
            last_episode_reward=last_score, last_episode_aux=last_aux, total_aux=aux_total,
            warmup_trace_sha256=trace.hexdigest() if step >= cfg["min_replay_size"] else None,
            error=error, updated_at_utc=datetime.now(timezone.utc).isoformat()))
    def save():
        temporary = args.output / "model.tmp"
        torch.save(dict(policy_net=policy.state_dict(), target_net=target.state_dict(),
            optimizer=optimizer.state_dict(), config=cfg, rng=rng(), epsilon=epsilon,
            step=source_step+step, source_step=source_step, trained_decisions=step,
            warmup_trace_sha256=trace.hexdigest()), temporary)
        temporary.replace(args.output / "model.pt")
    epsilon = float(start.get("epsilon", cfg["epsilon_start"]))
    progress()
    try:
        for step in range(1, args.decisions+1):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad(): action = int(policy(torch.from_numpy(state).unsqueeze(0).to(device)).argmax(1))
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state)
            episode_score += info["raw_reward"]; episode_aux += info["aux_reward"]; aux_total += info["aux_reward"]
            if step <= cfg["min_replay_size"]:
                trace.update(state.tobytes()); trace.update(bytes([action])); trace.update(repr(info["raw_reward"]).encode())
            transition = (state, action, reward, next_state, info["learning_terminal"])
            if len(replay) < cfg["buffer_capacity"]: replay.append(transition)
            else: replay[position] = transition
            position = (position+1) % cfg["buffer_capacity"]
            state = next_state
            if terminated or truncated:
                state, _ = env.reset(); state = np.array(state)
                last_score, last_aux, episode_score, episode_aux = episode_score, episode_aux, 0., 0.
            absolute = source_step+step
            epsilon = max(cfg["epsilon_end"], cfg["epsilon_start"] -
                          (cfg["epsilon_start"]-cfg["epsilon_end"])*absolute/cfg["epsilon_decay_steps"])
            if len(replay) >= cfg["min_replay_size"] and absolute % cfg["train_freq"] == 0:
                states, actions, rewards, next_states, terminal = zip(*random.sample(replay, cfg["batch_size"]))
                states = torch.as_tensor(np.stack(states), device=device)
                next_states = torch.as_tensor(np.stack(next_states), device=device)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                terminal = torch.tensor(terminal, dtype=torch.bool, device=device).unsqueeze(1)
                q = policy(states)
                expected = compute_targets(policy, target, next_states, rewards, terminal, cfg["gamma"], cfg["algorithm"])
                loss = nn.functional.smooth_l1_loss(q.gather(1, actions), expected)
                optimizer.zero_grad(); loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(policy.parameters(), 10.))
                if not torch.isfinite(loss) or not math.isfinite(grad_norm): raise FloatingPointError("Non-finite update")
                optimizer.step(); updates += 1
                loss_value, q_std = float(loss.detach()), float(q.detach().std())
                if step % 1000 == 0:
                    with torch.no_grad(): feature_std = float(policy.conv(states.float()/255.).std(dim=0).mean())
                    collapsed_checks = collapsed_checks+1 if feature_std < 1e-10 else 0
                    if collapsed_checks >= 3: raise RuntimeError("Image features collapsed")
            if absolute % cfg["target_update_freq"] == 0: target.load_state_dict(policy.state_dict())
            if step % 250_000 == 0: save()
            if args.eval_every and step % args.eval_every == 0:
                if __package__: from .evaluate import evaluate
                else: from evaluate import evaluate
                save(); status = "evaluating"; progress()
                checkpoint = args.output / f"model-{absolute:09d}.pt"
                shutil.copyfile(args.output / "model.pt", checkpoint)
                training_rng = rng()
                try:
                    report = evaluate(checkpoint, args.output / f"eval-{absolute:09d}.json",
                                      args.eval_episodes, args.eval_seed, args.device)
                finally:
                    restore_rng(training_rng, device)
                metrics = args.output / "comparison.csv"
                row = dict(step=absolute, **report["summary"])
                with metrics.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if handle.tell() == 0: writer.writeheader()
                    writer.writerow(row)
                status = "running"; progress()
            if step % 1000 == 0:
                progress(); print(f"{args.reward} {step}/{args.decisions} loss={loss_value:.4f} raw={last_score}", flush=True)
        status = "completed"
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        save(); progress(); env.close()


if __name__ == "__main__":
    main()
