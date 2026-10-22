import torch
import torch.nn as nn
import gymnasium as gym
import ale_py

class BreakeoutBasicNetwork(nn.Module):
    def __init__(self, in_channels: int = 4, n_actions: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x.float() / 255.0))


# Alias for corrected spelling
BreakoutBasicNetwork = BreakeoutBasicNetwork


def make_breakout_env(render_mode=None):
    """
    Creates an ALE/Breakout-v5 environment with standard Atari preprocessing
    and 4-frame stacking so observations match (4, 84, 84).
    """
    gym.register_envs(ale_py)
    env = gym.make(
        "ALE/Breakout-v5",
        mode=0,
        difficulty=0,
        render_mode=render_mode,
        frameskip=1,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env