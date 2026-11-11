"""Numerical checks for the DQN and Double DQN bootstrap targets."""

import unittest

import torch
from torch import nn

from experiments.ConceptCheck.dqn import compute_td_targets


class TableNetwork(nn.Module):
    """Return explicit Q values so action-selection mistakes are visible."""

    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values[: states.shape[0]]


class ComputeTDTargetsTests(unittest.TestCase):
    def test_online_selection_changes_double_dqn_target(self):
        policy = TableNetwork([[10.0, 1.0], [0.0, 9.0]])
        target = TableNetwork([[1.0, 8.0], [6.0, 3.0]])
        next_states = torch.zeros(2, 1)
        rewards = torch.tensor([[2.0], [-1.0]])
        terminated = torch.zeros(2, 1)

        dqn = compute_td_targets(
            policy, target, next_states, rewards, terminated, 0.9, "dqn"
        )
        double_dqn = compute_td_targets(
            policy, target, next_states, rewards, terminated, 0.9, "double-dqn"
        )

        torch.testing.assert_close(dqn, torch.tensor([[9.2], [4.4]]))
        torch.testing.assert_close(double_dqn, torch.tensor([[2.9], [1.7]]))
        self.assertEqual(dqn.shape, (2, 1))
        self.assertEqual(double_dqn.shape, (2, 1))

    def test_algorithms_agree_when_greedy_actions_agree(self):
        policy = TableNetwork([[1.0, 5.0], [7.0, 2.0]])
        target = TableNetwork([[2.0, 4.0], [3.0, -1.0]])
        next_states = torch.zeros(2, 1)
        rewards = torch.tensor([[1.0], [-2.0]])
        terminated = torch.zeros(2, 1)

        for algorithm in ("dqn", "double-dqn"):
            with self.subTest(algorithm=algorithm):
                actual = compute_td_targets(
                    policy, target, next_states, rewards, terminated, 0.5,
                    algorithm,
                )
                torch.testing.assert_close(actual, torch.tensor([[3.0], [-0.5]]))

    def test_only_terminated_transitions_stop_bootstrapping(self):
        policy = TableNetwork([[0.0, 3.0], [0.0, 3.0]])
        target = TableNetwork([[1.0, 8.0], [1.0, 8.0]])
        next_states = torch.zeros(2, 1)
        rewards = torch.tensor([[2.0], [2.0]])
        terminated = torch.tensor([[1.0], [0.0]])

        for algorithm in ("dqn", "double-dqn"):
            with self.subTest(algorithm=algorithm):
                actual = compute_td_targets(
                    policy, target, next_states, rewards, terminated, 0.9,
                    algorithm,
                )
                torch.testing.assert_close(actual, torch.tensor([[2.0], [9.2]]))

    def test_targets_detach_networks_and_rewards_from_autograd(self):
        policy = TableNetwork([[10.0, 1.0]])
        target = TableNetwork([[1.0, 8.0]])
        rewards = torch.tensor([[2.0]], requires_grad=True)

        for algorithm in ("dqn", "double-dqn"):
            with self.subTest(algorithm=algorithm):
                actual = compute_td_targets(
                    policy, target, torch.zeros(1, 1), rewards,
                    torch.zeros(1, 1), 0.9, algorithm,
                )
                self.assertFalse(actual.requires_grad)
                self.assertIsNone(actual.grad_fn)
                self.assertIsNone(policy.values.grad)
                self.assertIsNone(target.values.grad)
                self.assertIsNone(rewards.grad)

    def test_default_algorithm_is_dqn_and_preserves_singleton_batch(self):
        actual = compute_td_targets(
            TableNetwork([[10.0, 1.0]]),
            TableNetwork([[1.0, 8.0]]),
            torch.zeros(1, 1),
            torch.tensor([[2.0]]),
            torch.zeros(1, 1),
            0.9,
        )

        self.assertEqual(actual.shape, (1, 1))
        torch.testing.assert_close(actual, torch.tensor([[9.2]]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_targets_preserve_device_and_mask_boolean_terminals(self):
        policy = TableNetwork([[10.0, 1.0], [0.0, 9.0], [10.0, 1.0]]).to("cuda")
        target = TableNetwork([[1.0, 8.0], [6.0, 3.0], [2.0, 7.0]]).to("cuda")
        next_states = torch.zeros(3, 1, device="cuda")
        rewards = torch.tensor(
            [[2.0], [-1.0], [5.0]], device="cuda", requires_grad=True,
        )
        terminated = torch.tensor(
            [[False], [False], [True]], dtype=torch.bool, device="cuda",
        )
        expected_by_algorithm = {
            "dqn": [[9.2], [4.4], [5.0]],
            "double-dqn": [[2.9], [1.7], [5.0]],
        }

        for algorithm, expected in expected_by_algorithm.items():
            with self.subTest(algorithm=algorithm):
                actual = compute_td_targets(
                    policy, target, next_states, rewards, terminated, 0.9,
                    algorithm,
                )
                torch.testing.assert_close(
                    actual, torch.tensor(expected, device="cuda"),
                )
                self.assertEqual(actual.device.type, "cuda")
                self.assertFalse(actual.requires_grad)
                self.assertIsNone(actual.grad_fn)

    def test_unknown_algorithm_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_td_targets(
                TableNetwork([[1.0, 2.0]]),
                TableNetwork([[1.0, 2.0]]),
                torch.zeros(1, 1),
                torch.zeros(1, 1),
                torch.zeros(1, 1),
                0.9,
                "unsupported",
            )


if __name__ == "__main__":
    unittest.main()
