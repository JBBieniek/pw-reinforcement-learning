"""TD targets shared by DQN and Double DQN."""

import torch
from torch import nn


@torch.no_grad()
def compute_td_targets(
    policy_net: nn.Module,
    target_net: nn.Module,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    algorithm: str = "dqn",
) -> torch.Tensor:
    """Return detached [batch, 1] targets, masking true terminals only.

    Time-limit truncations still bootstrap from their final observation.
    Double DQN selects actions with the online network and evaluates them
    with the target network (van Hasselt et al., arXiv:1509.06461).
    """
    if algorithm not in ("dqn", "double-dqn"):
        raise ValueError(f"Unknown algorithm: {algorithm}")

    target_q_values = target_net(next_states)
    if algorithm == "double-dqn":
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_q_values.gather(1, next_actions)
    else:
        next_q = target_q_values.max(dim=1, keepdim=True).values

    return rewards + (1.0 - terminated.to(rewards.dtype)) * gamma * next_q
