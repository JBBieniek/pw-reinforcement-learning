import os
import time
import argparse
import numpy as np
import torch

from neural_network import BreakeoutBasicNetwork, make_breakout_env


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained DQN agent on Breakout")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "breakout_dqn.pth"),
        help="Path to the trained PyTorch model weights (.pth)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes to run")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "none"], help="Render mode")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum steps per episode")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS for human rendering playback")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on ('cpu', 'cuda')")
    return parser.parse_args()


def test():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    render_mode = "human" if args.render_mode == "human" else None
    env = make_breakout_env(render_mode=render_mode)
    n_actions = env.action_space.n

    model = BreakeoutBasicNetwork(in_channels=4, n_actions=n_actions).to(device)

    if os.path.exists(args.model_path):
        print(f"Loading trained weights from: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"[WARNING] Model checkpoint '{args.model_path}' not found!")
        print("Running with untrained / randomly initialized model for demonstration.")
        print("To test a trained model, run 'python train.py' first to generate weights.\n")

    model.eval()

    episode_rewards = []
    frame_delay = (1.0 / args.fps) if (render_mode == "human" and args.fps > 0) else 0.0

    print(f"Starting evaluation for {args.episodes} episode(s)...")

    try:
        for ep in range(1, args.episodes + 1):
            state, _ = env.reset()
            state = np.array(state)

            # In Atari Breakout, action 1 is FIRE to launch the initial ball
            next_state, init_reward, terminated, truncated, _ = env.step(1)
            state = np.array(next_state)

            total_reward = init_reward
            steps = 0
            done = terminated or truncated

            while not done and steps < args.max_steps:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action = q_values.argmax(dim=1).item()

                next_state, reward, terminated, truncated, _ = env.step(action)
                state = np.array(next_state)
                total_reward += reward
                steps += 1
                done = terminated or truncated

                if frame_delay > 0:
                    time.sleep(frame_delay)

            episode_rewards.append(total_reward)
            print(f"Episode {ep:2d}/{args.episodes}: Total Reward = {total_reward:5.1f} | Steps = {steps}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        env.close()

    if episode_rewards:
        print("\n" + "=" * 40)
        print("EVALUATION SUMMARY")
        print("=" * 40)
        print(f"Episodes played: {len(episode_rewards)}")
        print(f"Mean Reward:     {np.mean(episode_rewards):.2f}")
        print(f"Min Reward:      {np.min(episode_rewards):.2f}")
        print(f"Max Reward:      {np.max(episode_rewards):.2f}")
        print("=" * 40)


if __name__ == "__main__":
    test()
