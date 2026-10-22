import os
import random
from collections import deque
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from neural_network import BreakeoutBasicNetwork, make_breakout_env


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN agent on Breakout")
    parser.add_argument("--steps", type=int, default=100_000, help="Total training environment steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for gradient descent")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final exploration rate")
    parser.add_argument("--epsilon-decay-steps", type=int, default=80_000, help="Steps over which epsilon decays")
    parser.add_argument("--buffer-capacity", type=int, default=50_000, help="Replay buffer max capacity")
    parser.add_argument("--min-replay-size", type=int, default=5_000, help="Minimum transitions before training")
    parser.add_argument("--target-update-freq", type=int, default=1_000, help="Frequency of target net update (steps)")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint save frequency (steps)")
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "breakout_dqn.pth"),
        help="Path where model weights will be saved",
    )
    parser.add_argument("--render", action="store_true", help="Render environment window during training (slows down training)")
    return parser.parse_args()


def train():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    render_mode = "human" if args.render else None
    env = make_breakout_env(render_mode=render_mode)

    n_actions = env.action_space.n
    policy_net = BreakeoutBasicNetwork(in_channels=4, n_actions=n_actions).to(device)
    target_net = BreakeoutBasicNetwork(in_channels=4, n_actions=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    replay_buffer = deque(maxlen=args.buffer_capacity)

    state, _ = env.reset()
    state = np.array(state)

    epsilon = args.epsilon_start
    recent_loss = 0.0
    episode_reward = 0.0
    last_episode_reward = 0.0
    episodes_completed = 0

    pbar = tqdm(range(1, args.steps + 1), desc="Training DQN", unit="step")

    try:
        for step in pbar:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated

            episode_reward += reward
            replay_buffer.append((state, action, reward, next_state, done))

            if done:
                state, _ = env.reset()
                state = np.array(state)
                last_episode_reward = episode_reward
                episode_reward = 0.0
                episodes_completed += 1
            else:
                state = next_state

            # Epsilon linear decay
            if step <= args.epsilon_decay_steps:
                epsilon = args.epsilon_start - (args.epsilon_start - args.epsilon_end) * (step / args.epsilon_decay_steps)
            else:
                epsilon = args.epsilon_end

            # Gradient descent step
            if len(replay_buffer) >= args.min_replay_size:
                batch = random.sample(replay_buffer, args.batch_size)
                b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

                b_states = torch.from_numpy(np.array(b_states)).to(device)
                b_actions = torch.tensor(b_actions, dtype=torch.long, device=device).unsqueeze(1)
                b_rewards = torch.tensor(b_rewards, dtype=torch.float32, device=device).unsqueeze(1)
                b_next_states = torch.from_numpy(np.array(b_next_states)).to(device)
                b_dones = torch.tensor(b_dones, dtype=torch.float32, device=device).unsqueeze(1)

                current_q = policy_net(b_states).gather(1, b_actions)

                with torch.no_grad():
                    max_next_q = target_net(b_next_states).max(1)[0].unsqueeze(1)
                    target_q = b_rewards + (1.0 - b_dones) * args.gamma * max_next_q

                loss = loss_fn(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                recent_loss = loss.item()

            # Target network update
            if step % args.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Periodic checkpoint save
            if step % args.save_freq == 0:
                torch.save(policy_net.state_dict(), args.save_path)

            # Update progress bar metrics
            if step % 20 == 0 or step == args.steps:
                pbar.set_postfix({
                    "loss": f"{recent_loss:.4f}",
                    "eps": f"{epsilon:.3f}",
                    "buf": len(replay_buffer),
                    "last_ep_rew": last_episode_reward,
                    "episodes": episodes_completed,
                })

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
    finally:
        env.close()
        # Save final model
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(policy_net.state_dict(), args.save_path)
        print(f"Model successfully saved to: {args.save_path}")


if __name__ == "__main__":
    train()
