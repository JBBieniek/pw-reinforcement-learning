"""Regression checks for dead visual features and non-invasive diagnostics."""

import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from torch import nn

from experiments.ConceptCheck.health import (
    HealthMonitor, RepresentationCollapseError, inspect_network_health,
)
from experiments.ConceptCheck.neural_network import BreakoutBasicNetwork, configure_activation


class NegativeCNN(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(4, 2))
        with torch.no_grad():
            self.conv[0].weight.fill_(0.1)
            self.conv[0].bias.fill_(-2.0)
            self.fc[0].weight.fill_(1.0)
            self.fc[0].bias.zero_()
        configure_activation(self, activation)

    def forward(self, observations):
        return self.fc(self.conv(observations.float() / 255.0))


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.observations = torch.stack((torch.zeros(1, 2, 2), torch.full((1, 2, 2), 255))).to(torch.uint8)

    def backward(self, model):
        nn.functional.mse_loss(model(self.observations), torch.ones(2, 2)).backward()

    def test_leaky_activation_restores_visual_variation_and_gradient_through_negative_features(self):
        relu = NegativeCNN("relu")
        leaky = NegativeCNN("leaky-relu")
        leaky.load_state_dict(relu.state_dict())
        self.backward(relu)
        self.backward(leaky)
        old = inspect_network_health(relu, self.observations)
        repaired = inspect_network_health(leaky, self.observations)
        self.assertTrue(old["features_all_zero"])
        self.assertEqual(old["feature_std_across_states"], 0.0)
        self.assertEqual(old["conv_gradient_norm"], 0.0)
        self.assertTrue(old["conv_gradients_all_zero"])
        self.assertFalse(repaired["features_all_zero"])
        self.assertGreater(repaired["feature_std_across_states"], 0.0)
        self.assertGreater(repaired["q_std_across_states"], 0.0)
        self.assertGreater(repaired["conv_gradient_norm"], 0.0)

    def test_diagnostic_preserves_rng_weights_gradients_and_module_modes(self):
        model = NegativeCNN("leaky-relu")
        model.fc.eval()
        self.backward(model)
        modes = [module.training for module in model.modules()]
        parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        report = inspect_network_health(model, self.observations)
        self.assertTrue(report["finite"])
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameters[name], parameter))
            self.assertTrue(torch.equal(gradients[name], parameter.grad))

    def test_monitor_requires_repeated_zero_features_and_gradients_and_saves_evidence(self):
        model = NegativeCNN()
        self.backward(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.health.json"
            monitor = HealthMonitor(path, patience=3)
            monitor.check(model, self.observations, 1000)
            monitor.check(model, self.observations, 2000)
            with self.assertRaisesRegex(RepresentationCollapseError, "step 3000"):
                monitor.check(model, self.observations, 3000)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["consecutive_collapsed_checks"], 3)
            self.assertEqual(saved["step"], 3000)
            self.assertTrue(saved["collapsed"])
            self.assertEqual(len(monitor.history_path.read_text().splitlines()), 3)

    def test_constant_greedy_action_does_not_trigger_collapse(self):
        model = NegativeCNN("leaky-relu")
        self.backward(model)
        with tempfile.TemporaryDirectory() as directory:
            monitor = HealthMonitor(Path(directory) / "health.json", patience=1)
            report = monitor.check(model, self.observations, 1000)
            self.assertEqual(sum(count > 0 for count in report["greedy_action_counts"].values()), 1)
            self.assertFalse(report["collapsed"])
            self.assertEqual(report["collapse_reasons"], [])

    def test_exactly_constant_q_values_trigger_collapse_even_with_live_features(self):
        model = NegativeCNN("leaky-relu")
        with torch.no_grad():
            model.fc[0].weight.zero_()
            model.fc[0].bias.copy_(torch.tensor([3.0, 4.0]))
        self.backward(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            monitor = HealthMonitor(path, patience=3)
            first = monitor.check(model, self.observations, 1000)
            self.assertFalse(first["features_all_zero"])
            self.assertGreater(first["feature_std_across_states"], 0.0)
            self.assertTrue(first["conv_gradients_present"])
            self.assertEqual(first["q_std_across_states"], 0.0)
            self.assertEqual(first["collapse_reasons"], ["constant_q_values"])
            monitor.check(model, self.observations, 2000)
            with self.assertRaisesRegex(RepresentationCollapseError, "constant_q_values"):
                monitor.check(model, self.observations, 3000)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["consecutive_collapsed_checks"], 3)
            self.assertEqual(saved["collapse_reasons"], ["constant_q_values"])

    def test_exactly_constant_nonzero_features_trigger_collapse_with_nonzero_gradients(self):
        model = NegativeCNN()
        with torch.no_grad():
            model.conv[0].weight.zero_()
            model.conv[0].bias.fill_(1.0)
        self.backward(model)
        with tempfile.TemporaryDirectory() as directory:
            monitor = HealthMonitor(Path(directory) / "health.json", patience=3)
            first = monitor.check(model, self.observations, 1000)
            self.assertFalse(first["features_all_zero"])
            self.assertEqual(first["feature_std_across_states"], 0.0)
            self.assertGreater(first["conv_gradient_norm"], 0.0)
            self.assertEqual(first["collapse_reasons"], ["constant_nonzero_features", "constant_q_values"])
            monitor.check(model, self.observations, 2000)
            with self.assertRaisesRegex(RepresentationCollapseError, "constant_nonzero_features"):
                monitor.check(model, self.observations, 3000)

    def test_no_false_failure_before_gradients_or_without_diverse_observations(self):
        dead_features = NegativeCNN()
        constant_q = NegativeCNN("leaky-relu")
        with torch.no_grad():
            constant_q.fc[0].weight.zero_()
        for model in (dead_features, constant_q):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                monitor = HealthMonitor(Path(directory) / "health.json", patience=1)
                before_update = monitor.check(model, self.observations, 1000)
                self.assertFalse(before_update["conv_gradients_present"])
                self.assertFalse(before_update["collapsed"])
                self.assertEqual(before_update["collapse_reasons"], [])
                self.backward(model)
                repeated_state = self.observations[:1].expand(2, -1, -1, -1)
                report = monitor.check(model, repeated_state, 2000)
                self.assertFalse(report["diverse_observations"])
                self.assertFalse(report["collapsed"])
                self.assertEqual(report["collapse_reasons"], [])

    def test_monitor_counter_can_be_restored(self):
        model = NegativeCNN()
        self.backward(model)
        with tempfile.TemporaryDirectory() as directory:
            monitor = HealthMonitor(Path(directory) / "health.json", patience=2)
            monitor.check(model, self.observations, 1000)
            restored = HealthMonitor(Path(directory) / "restored.health.json", patience=2)
            restored.set_state(monitor.get_state())
            with self.assertRaises(RepresentationCollapseError):
                restored.check(model, self.observations, 2000)

    def test_non_finite_features_are_saved_as_a_failure_without_invalid_json(self):
        model = NegativeCNN("leaky-relu")
        with torch.no_grad():
            model.conv[0].weight.fill_(float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            monitor = HealthMonitor(path)
            with self.assertRaisesRegex(RepresentationCollapseError, "Non-finite"):
                monitor.check(model, self.observations, 1000)
            saved = json.loads(path.read_text())
            self.assertFalse(saved["finite"])
            self.assertEqual(saved["status"], "non_finite")
            self.assertIsNone(saved["feature_std_across_states"])

    def test_legacy_constructor_and_state_dict_compatibility_are_preserved(self):
        model = BreakoutBasicNetwork()
        self.assertTrue(any(isinstance(module, nn.ReLU) for module in model.modules()))
        weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        result = configure_activation(model, "leaky-relu")
        self.assertIs(result, model)
        self.assertFalse(any(isinstance(module, nn.ReLU) for module in model.modules()))
        self.assertEqual(sum(isinstance(module, nn.LeakyReLU) for module in model.modules()), 4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(weights.keys(), model.state_dict().keys())
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(weights[name], tensor))
        configure_activation(model, "relu")
        self.assertEqual(sum(isinstance(module, nn.ReLU) for module in model.modules()), 4)
        with self.assertRaises(ValueError):
            configure_activation(model, "invalid")

    def test_unsupported_stateful_models_are_not_executed(self):
        model = NegativeCNN()
        model.conv.insert(1, nn.Dropout())
        rng = torch.get_rng_state().clone()
        self.assertFalse(inspect_network_health(model, self.observations)["supported"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(inspect_network_health(nn.Linear(1, 2), torch.zeros(2, 1))["supported"])


if __name__ == "__main__":
    unittest.main()
