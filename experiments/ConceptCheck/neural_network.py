import torch
import torch.nn as nn
import gymnasium as gym
import ale_py


def configure_activation(model: nn.Module, activation: str = "relu") -> nn.Module:
    """Choose the same activation for both algorithms without changing weights.

    The legacy ReLU network remains the default when reading old policy files.
    Fresh training can use LeakyReLU so negative activations retain a gradient.
    Activations have no parameters, so existing state_dict keys stay compatible;
    callers must nevertheless preserve the activation name in model metadata.
    """
    if activation not in ("relu", "leaky-relu"):
        raise ValueError(f"Unknown activation: {activation!r}")
    for name, child in list(model.named_children()):
        if isinstance(child, (nn.ReLU, nn.LeakyReLU)):
            replacement = nn.ReLU() if activation == "relu" else nn.LeakyReLU(negative_slope=0.01)
            replacement.train(child.training)
            setattr(model, name, replacement)
        else:
            configure_activation(child, activation)
    return model


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


def make_breakout_env(
    render_mode=None, *, mode=0, difficulty=0, frame_skip=4,
    sticky_action_probability=0.25, reward_shaping="none", bounce_bonus=1.0,
    rally_bonus_step=0.2, rally_bonus_max_multiplier=2.0,
):
    """
    Creates an ALE/Breakout-v5 environment with standard Atari preprocessing
    and 4-frame stacking so observations match (4, 84, 84).
    """
    if reward_shaping not in ("none", "rally"):
        raise ValueError(f"Unknown reward shaping: {reward_shaping!r}")
    if reward_shaping == "rally" and mode not in (0, 4, 12, 32, 36, 44):
        raise ValueError("Rally rewards support the selected modes 0, 4, 12, 32, 36, 44")
    gym.register_envs(ale_py)
    env = gym.make(
        "ALE/Breakout-v5",
        mode=mode,
        difficulty=difficulty,
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=sticky_action_probability,
    )
    if reward_shaping == "rally":
        # Observe native RGB frames before AtariPreprocessing repeats actions.
        # The wrapper preserves game rewards; training adds its separate bonus.
        from experiments.ConceptCheck.rewards import RallyRewardTracker
        env = RallyRewardTracker(
            env, bounce_bonus=bounce_bonus, rally_bonus_step=rally_bonus_step,
            rally_bonus_max_multiplier=rally_bonus_max_multiplier,
        )
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env