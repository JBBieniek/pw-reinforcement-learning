import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make(
    "ALE/Breakout-v5",
    mode=0,
    difficulty=0,
    render_mode="human",
)

obs, info = env.reset(seed=42)

# FIRE – uruchomienie piłki
env.step(1)

for _ in range(5000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        env.step(1)

env.close()
