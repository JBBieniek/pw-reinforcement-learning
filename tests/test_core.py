"""Small standalone contracts; run with python -m unittest discover -s tests."""
import unittest
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from experiments.ConceptCheck.rewards import NativeSignals, RewardWrapper, extra_reward, landing_potential
from experiments.ConceptCheck.train import compute_targets


def frame(y=177):
    image = np.zeros((210, 160, 3), dtype=np.uint8)
    image[189:193, 64:80] = (200, 72, 72)
    if y is not None:
        image[y:y+4, 70:72] = (200, 72, 72)
    return image


class Native(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (210,160,3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(4)
    def reset(self, **kwargs):
        self.position = 0
        return frame(), {"lives": 5}
    def step(self, action):
        y, reward = [(179,0),(181,0),(183,0),(181,0),(170,1),(160,4),(150,7),(None,0)][self.position]
        self.position += 1
        return frame(y), reward, False, False, {"lives": 4 if y is None else 5}


class CoreTests(unittest.TestCase):
    def test_double_dqn_selection_and_terminal_mask(self):
        policy, target = nn.Linear(1, 2), nn.Linear(1, 2)
        with torch.no_grad():
            policy.weight.zero_(); target.weight.zero_()
            policy.bias.copy_(torch.tensor([9., 2.])); target.bias.copy_(torch.tensor([1., 4.]))
        state, reward, terminal = torch.zeros(2, 1), torch.tensor([[2.], [3.]]), torch.tensor([[False], [True]])
        ddqn = compute_targets(policy, target, state, reward, terminal, .5)
        dqn = compute_targets(policy, target, state, reward, terminal, .5, "dqn")
        torch.testing.assert_close(ddqn, torch.tensor([[2.5], [3.]]))
        torch.testing.assert_close(dqn, torch.tensor([[4.], [3.]]))
        self.assertFalse(ddqn.requires_grad)

    def test_bounce_cap_raw_points_and_life_reset(self):
        signals = NativeSignals(Native())
        env = RewardWrapper(signals, signals, "small-bounce", .995)
        env.reset()
        rewards = [env.step(0)[1] for _ in range(7)]
        self.assertEqual(signals.bounces, 1)
        self.assertAlmostEqual(signals.total_bonus, 2)
        self.assertEqual(env.raw_score, 12)
        self.assertAlmostEqual(sum(rewards), 12.1)
        _, reward, _, _, info = env.step(0)
        self.assertTrue(info["learning_terminal"])
        self.assertEqual((reward, signals.phi, signals.history), (0,0,[]))

    def test_potential_targets_catch_band_and_reflects_wall(self):
        history = [(t,140.5+2*t,131.5+t) for t in range(5)]
        self.assertEqual(landing_potential(history,(148,134,149,137),(42,189,57,192)),1)
        stationary = [(t,70.5,130.5) for t in range(7)]
        self.assertEqual(landing_potential(stationary,(70,129,71,132),(64,189,79,192)),0)

    def test_shaping_telescopes_and_never_pays_constant_positive_for_waiting(self):
        gamma, phi = .995, [0,.7,.9,.4]
        r = [extra_reward("landing",0,phi[i],phi[i+1],i==2,gamma) for i in range(3)]
        self.assertAlmostEqual(sum(gamma**i*x for i,x in enumerate(r)),0,places=12)
        self.assertLessEqual(extra_reward("landing",0,1,1,False,gamma),0)
        self.assertEqual(extra_reward("raw",2,0,1,False,gamma),0)


if __name__ == "__main__":
    unittest.main()
