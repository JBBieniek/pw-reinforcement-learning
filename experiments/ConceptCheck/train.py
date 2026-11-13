import os
import random
import secrets
import sys
from pathlib import Path
from shutil import copyfile
from collections import deque
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

if __package__:
    from .dqn import compute_td_targets
    from .neural_network import BreakoutBasicNetwork, configure_activation, make_breakout_env
    from .runtime import configure_determinism
    from .health import HealthMonitor, RepresentationCollapseError
    from .run_logging import TrainingRun
    from .pacing import StepPacer
    from .replay import FrameReplayBuffer
    from .checkpoints import (load_training_checkpoint, restore_training_checkpoint,
                              save_training_checkpoint, training_checkpoint_path)
else:
    # Keep pickled replay classes importable from module and script entry points.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dqn import compute_td_targets
    from neural_network import BreakoutBasicNetwork, configure_activation, make_breakout_env
    from runtime import configure_determinism
    from health import HealthMonitor, RepresentationCollapseError
    from run_logging import TrainingRun
    from pacing import StepPacer
    from experiments.ConceptCheck.replay import FrameReplayBuffer
    from checkpoints import (load_training_checkpoint, restore_training_checkpoint,
                             save_training_checkpoint, training_checkpoint_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DQN or Double DQN on Breakout", allow_abbrev=False)
    parser.add_argument("--algorithm", choices=["dqn", "double-dqn"], default="dqn", help="TD target calculation")
    parser.add_argument("--activation", choices=["relu", "leaky-relu"], default="leaky-relu", help="CNN activation (saved with weights)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True, help="Use reproducible GPU operations on this software/hardware stack")
    parser.add_argument("--health-freq", type=int, default=1_000, help="Steps between CNN health checks (0: disabled)")
    parser.add_argument("--health-patience", type=int, default=3, help="Consecutive dead-feature checks before stopping")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: generate and save one for this run)")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device, e.g. cpu or cuda (default: auto)")
    parser.add_argument("--torch-threads", type=int, default=2, help="PyTorch CPU worker threads")
    parser.add_argument("--max-steps-per-second", type=float, default=0.0, help="Soft decision-rate limit to leave compute headroom (0: unlimited)")
    parser.add_argument("--mode", type=int, default=0, help="Breakout game mode")
    parser.add_argument("--difficulty", type=int, choices=[0, 1], default=0, help="Breakout difficulty")
    parser.add_argument("--frame-skip", type=int, choices=[2, 4, 8], default=4, help="Emulator frames per agent action")
    parser.add_argument("--sticky-action-probability", type=float, choices=[0.0, 0.25], default=0.25, help="Probability of repeating the previous action")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training environment steps")
    parser.add_argument("--stop-at", type=int, default=None, help="Pause at this absolute step while keeping --steps as the overall target")
    parser.add_argument("--resume", type=str, default=None, help="Trusted local .training.pt snapshot to continue")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000, help="Steps between full resumable snapshots and milestone weights")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Full snapshot path (default: <save-path>.training.pt)")
    parser.add_argument("--train-freq", type=int, default=4, help="Environment steps between gradient updates")
    parser.add_argument("--log-freq", type=int, default=1_000, help="Environment steps between progress and metrics writes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for gradient descent")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.1, help="Final exploration rate")
    parser.add_argument("--epsilon-decay-steps", type=int, default=250_000, help="Steps over which epsilon decays")
    parser.add_argument("--buffer-capacity", type=int, default=50_000, help="Replay buffer max capacity")
    parser.add_argument("--replay-storage", choices=["deque", "frames"], default="deque", help="Uniform replay storage; frames shares overlapping observation frames")
    parser.add_argument("--terminal-on-life-loss", action=argparse.BooleanOptionalAction, default=False, help="Mask the training TD bootstrap after a lost life, without resetting the full game")
    parser.add_argument("--min-replay-size", type=int, default=20_000, help="Minimum transitions before training")
    parser.add_argument("--target-update-freq", type=int, default=10_000, help="Frequency of target net update (steps)")
    parser.add_argument("--save-freq", type=int, default=25_000, help="Checkpoint save frequency (steps)")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Weights path (default: breakout_<algorithm>.pth next to this script)",
    )
    parser.add_argument("--render", action="store_true", help="Render environment window during training (slows down training)")
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments)
    supplied = {argument.split("=", 1)[0] for argument in arguments if argument.startswith("--")}
    resume_state = None
    if args.resume:
        try:
            resume_state = load_training_checkpoint(args.resume)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        saved = resume_state["config"]
        fixed = (
            "algorithm", "seed", "mode", "difficulty", "frame_skip",
            "sticky_action_probability", "train_freq", "batch_size", "lr", "gamma",
            "epsilon_start", "epsilon_end", "epsilon_decay_steps", "buffer_capacity",
            "min_replay_size", "target_update_freq", "activation", "deterministic",
            "replay_storage", "terminal_on_life_loss",
        )
        legacy_defaults = {"activation": "relu", "deterministic": False,
                           "replay_storage": "deque", "terminal_on_life_loss": False}
        for name in fixed:
            option = "--" + name.replace("_", "-")
            value = saved[name] if name in saved else legacy_defaults[name]
            explicit = option in supplied or (name in ("deterministic", "terminal_on_life_loss")
                                              and "--no-" + name.replace("_", "-") in supplied)
            if explicit and getattr(args, name) != value:
                parser.error(f"{option} differs from the training checkpoint ({value!r})")
            setattr(args, name, value)
        for name in ("steps", "save_path", "save_freq", "log_freq", "checkpoint_freq", "checkpoint_path", "max_steps_per_second", "health_freq", "health_patience"):
            if "--" + name.replace("_", "-") not in supplied and name in saved:
                setattr(args, name, saved[name])
        if "--save-path" in supplied and "--checkpoint-path" not in supplied:
            args.checkpoint_path = None
    if args.seed is None:
        args.seed = secrets.randbits(32)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    for name in ("steps", "batch_size", "epsilon_decay_steps", "buffer_capacity", "min_replay_size", "target_update_freq", "save_freq", "train_freq", "log_freq", "torch_threads", "checkpoint_freq", "health_patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.health_freq < 0:
        parser.error("--health-freq must be non-negative")
    if not args.batch_size <= args.min_replay_size <= args.buffer_capacity:
        parser.error("require --batch-size <= --min-replay-size <= --buffer-capacity")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be between 0 and 1")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        parser.error("require 0 <= --epsilon-end <= --epsilon-start <= 1")
    if not args.lr > 0.0 or not np.isfinite(args.lr):
        parser.error("--lr must be finite and positive")
    if not np.isfinite(args.max_steps_per_second) or args.max_steps_per_second < 0:
        parser.error("--max-steps-per-second must be finite and non-negative")
    if args.save_path is None:
        filename = f"breakout_{args.algorithm.replace('-', '_')}.pth"
        args.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    args.save_path = os.path.abspath(args.save_path)
    args.checkpoint_path = os.path.abspath(args.checkpoint_path or training_checkpoint_path(args.save_path))
    if args.checkpoint_path == args.save_path:
        parser.error("--checkpoint-path and --save-path must be different files")
    start_step = resume_state["step"] if resume_state else 0
    if args.steps <= start_step:
        parser.error("--steps must exceed the checkpoint's completed step")
    if args.stop_at is not None and not start_step < args.stop_at <= args.steps:
        parser.error("--stop-at must exceed the saved step and cannot exceed --steps")
    args._resume_state = resume_state
    return args


def train(argv=None):
    args = parse_args(argv)
    resume_state = vars(args).pop("_resume_state")

    configure_determinism(args.deterministic)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    algorithm_name = "Double DQN" if args.algorithm == "double-dqn" else "DQN"
    print(f"Algorithm: {algorithm_name} | Device: {device} | Seed: {args.seed} | Activation: {args.activation} | Deterministic: {args.deterministic}")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    render_mode = "human" if args.render else None
    env = make_breakout_env(
        render_mode=render_mode,
        mode=args.mode,
        difficulty=args.difficulty,
        frame_skip=args.frame_skip,
        sticky_action_probability=args.sticky_action_probability,
    )
    env.action_space.seed(args.seed)

    n_actions = env.action_space.n
    policy_net = configure_activation(BreakoutBasicNetwork(in_channels=4, n_actions=n_actions), args.activation).to(device)
    target_net = configure_activation(BreakoutBasicNetwork(in_channels=4, n_actions=n_actions), args.activation).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    target_net.requires_grad_(False)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    replay_buffer = (FrameReplayBuffer(args.buffer_capacity) if args.replay_storage == "frames"
                     else deque(maxlen=args.buffer_capacity))

    state, reset_info = env.reset(seed=args.seed)
    state = np.array(state)
    previous_lives = reset_info.get("lives")

    epsilon = args.epsilon_start
    recent_loss = 0.0
    episode_reward = 0.0
    last_episode_reward = 0.0
    episodes_completed = 0

    step = 0
    elapsed_seconds = 0.0
    health = HealthMonitor(Path(args.save_path).with_suffix(".health.json"), patience=args.health_patience)
    if resume_state:
        restored = restore_training_checkpoint(
            resume_state, policy_net=policy_net, target_net=target_net,
            optimizer=optimizer, env=env, device=device,
        )
        replay_buffer = restored["replay_buffer"]
        state = restored["state"]
        step = restored["step"]
        epsilon = restored["epsilon"]
        episode_reward = restored["episode_reward"]
        last_episode_reward = restored["last_episode_reward"]
        episodes_completed = restored["episodes_completed"]
        recent_loss = restored["recent_loss"]
        elapsed_seconds = restored.get("elapsed_seconds", 0.0)
        previous_lives = restored.get("previous_lives")
        health.set_state(restored.get("health_state", {}))
        print(f"Resumed step {step}; replay={len(replay_buffer)}; "
              f"game restored={restored['environment_restored']}; "
              f"exact trajectory={restored['environment_restore_exact']}")
        del restored, resume_state
    if args.terminal_on_life_loss and previous_lives is None:
        env.close()
        raise ValueError("--terminal-on-life-loss requires the environment's lives counter")
    run = TrainingRun(args, device, start_step=step, elapsed_seconds=elapsed_seconds)
    end_step = min(args.steps, args.stop_at) if args.stop_at is not None else args.steps
    status = "paused" if end_step < args.steps else "completed"
    error = None
    step_in_progress = False
    last_full_step = None
    last_training_states = None
    run.record(step, recent_loss, epsilon, len(replay_buffer), episodes_completed, last_episode_reward)
    pbar = tqdm(range(step + 1, end_step + 1), desc=f"Training {algorithm_name}", unit="step", disable=not os.isatty(2))
    pacer = StepPacer(args.max_steps_per_second)

    def save_weights(path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(policy_net.state_dict(), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save_full_state():
        nonlocal last_full_step
        save_training_checkpoint(
            args.checkpoint_path, args=args, policy_net=policy_net,
            target_net=target_net, optimizer=optimizer, replay_buffer=replay_buffer,
            state=state, step=step, epsilon=epsilon, episode_reward=episode_reward,
            last_episode_reward=last_episode_reward, episodes_completed=episodes_completed,
            recent_loss=recent_loss, env=env, elapsed_seconds=run.elapsed_seconds(),
            health_state=health.get_state(),
            previous_lives=previous_lives,
        )
        milestone = Path(args.save_path).with_name(f"{Path(args.save_path).stem}.step-{step:09d}.pth")
        save_weights(milestone)
        copyfile(Path(args.save_path).with_suffix(".config.json"), milestone.with_suffix(".config.json"))
        last_full_step = step

    try:
        for next_step in pbar:
            step_in_progress = True
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated
            lives = info.get("lives")
            if args.terminal_on_life_loss and lives is None:
                raise ValueError("The environment stopped reporting its lives counter")
            life_lost = bool(args.terminal_on_life_loss and lives < previous_lives)
            learning_terminal = bool(terminated or life_lost)

            episode_reward += reward
            # Truncation resets the environment but must not mask bootstrapping.
            # The opt-in life-loss variant changes only this TD mask. Scores,
            # game resets and observation stacking still cover the full game.
            replay_buffer.append((state, action, reward, next_state, learning_terminal))

            if done:
                state, reset_info = env.reset()
                state = np.array(state)
                previous_lives = reset_info.get("lives")
                last_episode_reward = episode_reward
                episode_reward = 0.0
                episodes_completed += 1
            else:
                state = next_state
                previous_lives = lives

            # Epsilon linear decay
            if next_step <= args.epsilon_decay_steps:
                epsilon = args.epsilon_start - (args.epsilon_start - args.epsilon_end) * (next_step / args.epsilon_decay_steps)
            else:
                epsilon = args.epsilon_end

            # Gradient descent step
            if len(replay_buffer) >= args.min_replay_size and next_step % args.train_freq == 0:
                batch = (replay_buffer.sample(args.batch_size)
                         if isinstance(replay_buffer, FrameReplayBuffer)
                         else random.sample(replay_buffer, args.batch_size))
                b_states, b_actions, b_rewards, b_next_states, b_terminated = zip(*batch)

                b_states = torch.from_numpy(np.array(b_states)).to(device)
                b_actions = torch.tensor(b_actions, dtype=torch.long, device=device).unsqueeze(1)
                b_rewards = torch.tensor(b_rewards, dtype=torch.float32, device=device).unsqueeze(1)
                b_next_states = torch.from_numpy(np.array(b_next_states)).to(device)
                b_terminated = torch.tensor(b_terminated, dtype=torch.bool, device=device).unsqueeze(1)

                current_q = policy_net(b_states).gather(1, b_actions)

                target_q = compute_td_targets(
                    policy_net, target_net, b_next_states, b_rewards,
                    b_terminated, args.gamma, args.algorithm,
                )

                loss = loss_fn(current_q, target_q)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {next_step}")

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                recent_loss = loss.item()
                last_training_states = b_states

            # Target network update
            if next_step % args.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            step = next_step
            step_in_progress = False

            if args.health_freq and step % args.health_freq == 0 and last_training_states is not None:
                try:
                    health.check(policy_net, last_training_states, step)
                except RepresentationCollapseError:
                    # This completed step is safe to retain for diagnosis.
                    save_full_state()
                    raise

            # Periodic checkpoint save
            if step % args.save_freq == 0:
                save_weights(args.save_path)

            if step % args.checkpoint_freq == 0:
                save_full_state()

            if step % args.log_freq == 0:
                run.record(step, recent_loss, epsilon, len(replay_buffer), episodes_completed, last_episode_reward)
                print(f"step={step}/{args.steps} loss={recent_loss:.5f} epsilon={epsilon:.3f} episodes={episodes_completed} last_reward={last_episode_reward}", flush=True)

            # Update progress bar metrics
            if step % 20 == 0 or step == args.steps:
                pbar.set_postfix({
                    "loss": f"{recent_loss:.4f}",
                    "eps": f"{epsilon:.3f}",
                    "buf": len(replay_buffer),
                    "last_ep_rew": last_episode_reward,
                    "episodes": episodes_completed,
                })
            pacer.wait()

    except KeyboardInterrupt:
        status = "interrupted"
        print("\nTraining interrupted by user. Saving current model state...")
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        raise
    finally:
        try:
            save_weights(args.save_path)
            if status != "failed" and not step_in_progress and last_full_step != step:
                save_full_state()
            elif step_in_progress:
                print("Step was interrupted; resume from the previous full snapshot.")
            print(f"Model saved to: {args.save_path}")
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            raise
        finally:
            try:
                env.close()
            except Exception as exc:
                status = "failed"
                error = repr(exc)
                raise
            finally:
                run.record(step, recent_loss, epsilon, len(replay_buffer), episodes_completed, last_episode_reward, status, error)


if __name__ == "__main__":
    train()
