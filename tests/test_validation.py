"""Pure checks for the validation evidence, without launching training."""

from collections import deque
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.ConceptCheck.neural_network import BreakoutBasicNetwork, configure_activation
from experiments.ConceptCheck.validate_stability import (
    audit_validation_report, digest_value, inspect_fixed_probe, read_health, variation_thresholds,
)


class ValidationEvidenceTests(unittest.TestCase):
    def test_fingerprints_ignore_object_identity_and_detect_a_bit_change(self):
        array = np.arange(8, dtype=np.uint8)
        first = {"weights": torch.tensor([1.0, 2.0]), "replay": deque([(array, array)], maxlen=4)}
        copied = {"replay": deque([(array.copy(), array.copy())], maxlen=4),
                  "weights": torch.tensor([1.0, 2.0])}
        self.assertEqual(digest_value(first), digest_value(copied))
        copied["weights"][1] += 0.001
        self.assertNotEqual(digest_value(first), digest_value(copied))

    def test_generators_hash_state_without_advancing_it(self):
        first = np.random.default_rng(987123)
        second = np.random.default_rng(987123)
        self.assertEqual(digest_value(first), digest_value(second))
        self.assertEqual(first.integers(1000), second.integers(1000))
        first.integers(1000)
        self.assertNotEqual(digest_value(first), digest_value(second))

    def test_scalar_optimizer_counter_is_supported(self):
        self.assertEqual(digest_value(torch.tensor(15.0)), digest_value(torch.tensor(15.0)))
        self.assertNotEqual(digest_value(torch.tensor(15.0)), digest_value(torch.tensor(16.0)))

    def test_fixed_probe_detects_zero_representation_without_changing_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            model = BreakoutBasicNetwork()
            for parameter in model.parameters():
                parameter.data.zero_()
            torch.save(model.state_dict(), weights)
            observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
            observations[1] = 127
            rng = torch.get_rng_state().clone()
            result = inspect_fixed_probe(weights, observations)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["features_depend_on_observation"])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_health_requires_observation_dependence_and_live_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            health = {"step": 30000, "supported": True, "diverse_observations": True,
                      "finite": True, "features_all_zero": False,
                      "feature_std_across_states": 0.01,
                      "q_std_across_states": 0.001,
                      "conv_gradients_present": True, "conv_gradient_norm": 0.002}
            path = weights.with_suffix(".health.json")
            path.write_text(json.dumps(health), encoding="utf-8")
            self.assertTrue(read_health(weights, 30000)["passed"])
            health["feature_std_across_states"] = 0.0
            path.write_text(json.dumps(health), encoding="utf-8")
            self.assertFalse(read_health(weights, 30000)["passed"])
            health["feature_std_across_states"] = 0.01
            health["conv_gradient_norm"] = 0.0
            path.write_text(json.dumps(health), encoding="utf-8")
            self.assertFalse(read_health(weights, 30000)["passed"])
            health["conv_gradient_norm"] = 5e-9
            path.write_text(json.dumps(health), encoding="utf-8")
            self.assertFalse(read_health(weights, 30000)["passed"])

    def test_near_collapsed_leaky_network_fails_numeric_floors(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pth"
            model = configure_activation(BreakoutBasicNetwork(), "leaky-relu")
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.fill_(1e-6 if name.endswith("weight") else 0.0)
            torch.save(model.state_dict(), weights)
            observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
            observations[1] = 255
            result = inspect_fixed_probe(weights, observations)
            self.assertGreater(result["measurement"]["feature_std_across_states"], 0)
            self.assertGreater(result["measurement"]["q_std_across_states"], 0)
            self.assertFalse(result["checks"]["feature_variation_above_floor"])
            self.assertFalse(result["checks"]["q_variation_above_floor"])
            self.assertFalse(result["passed"])

    def test_relative_floor_is_calibrated_to_warmup(self):
        thresholds = variation_thresholds({"feature_std_across_states": 0.03,
                                            "q_std_across_states": 0.002})
        self.assertAlmostEqual(thresholds["feature_std_across_states"], 0.0003)
        self.assertAlmostEqual(thresholds["q_std_across_states"], 0.00002)
        self.assertEqual(thresholds["conv_gradient_norm"], 1e-8)

    def test_audit_refuses_a_live_report_without_changing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "validation.json"
            content = json.dumps({"status": "running", "passed": False})
            report_path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finish"):
                audit_validation_report(directory)
            self.assertEqual(report_path.read_text(encoding="utf-8"), content)

    def test_audit_can_reject_an_old_positive_report_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(381907)
                model = configure_activation(BreakoutBasicNetwork(), "leaky-relu")
            observations = np.zeros((2, 4, 84, 84), dtype=np.uint8)
            observations[1] = 255
            np.save(directory / "probe-observations.npy", observations)
            report = {"status": "completed", "passed": True, "stages": {},
                      "probe_observations": {"digest": digest_value(observations)}}
            for name in ("warmup", "replica-a", "replica-b", "stress"):
                weights = directory / name / "model.pth"
                weights.parent.mkdir()
                torch.save(model.state_dict(), weights)
                report["stages"][name] = {"weights": str(weights), "stop_at": 30000,
                                         "checkpoint": str(weights.with_suffix(".training.pt"))}
                health = {"step": 30000, "supported": True, "diverse_observations": True,
                          "finite": True, "features_all_zero": False,
                          "feature_std_across_states": 0.01, "q_std_across_states": 0.001,
                          "conv_gradients_present": True, "conv_gradient_norm": 0.002}
                weights.with_suffix(".health.json").write_text(json.dumps(health), encoding="utf-8")
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.fill_(1e-6 if name.endswith("weight") else 0.0)
            torch.save(model.state_dict(), directory / "stress" / "model.pth")
            (directory / "validation.json").write_text(json.dumps(report), encoding="utf-8")
            fingerprint = {"step": 30000, "replay_size": 30000,
                           "fingerprints": {"policy_net": "identical", "health_state": "identical"}}
            with patch("experiments.ConceptCheck.validate_stability.checkpoint_fingerprints",
                       return_value=fingerprint) as refresh:
                audited = audit_validation_report(directory)
            self.assertEqual(refresh.call_count, 2)
            self.assertTrue(audited["pre_calibration_passed"])
            self.assertFalse(audited["passed"])
            self.assertTrue(audited["calibrated_health"]["stages"]["replica-a"]["passed"])
            self.assertFalse(audited["calibrated_health"]["stages"]["stress"]["passed"])
            self.assertIn("health_state", audited["calibrated_health"]["exact_comparison_scope"]["components"])
            self.assertTrue(audited["repeatability"]["passed"])
            baseline = torch.load(directory / "probe-baseline.pth", weights_only=True)
            warmup = torch.load(directory / "warmup" / "model.pth", weights_only=True)
            self.assertEqual(digest_value(baseline), digest_value(warmup))

    def test_audit_refresh_rejects_watchdog_state_and_pacing_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "probe-observations.npy", np.zeros((2, 4, 84, 84), dtype=np.uint8))
            stages = {}
            for name in ("warmup", "replica-a", "replica-b", "paced-comparison", "stress"):
                weights = directory / name / "model.pth"
                weights.parent.mkdir()
                weights.write_bytes(b"fixture")
                stages[name] = {"weights": str(weights), "checkpoint": str(weights.with_suffix(".training.pt")),
                                "stop_at": 30000}
            report = {"status": "completed", "passed": True, "stages": stages}
            (directory / "validation.json").write_text(json.dumps(report), encoding="utf-8")
            fixed = {"passed": True, "measurement": {"feature_std_across_states": 0.03,
                                                       "q_std_across_states": 0.002}}
            first = {"step": 30000, "fingerprints": {"policy_net": "same", "health_state": "a"}}
            different = {"step": 30000, "fingerprints": {"policy_net": "same", "health_state": "b"}}
            with patch("experiments.ConceptCheck.validate_stability.inspect_fixed_probe", return_value=fixed), \
                    patch("experiments.ConceptCheck.validate_stability.read_health", return_value={"passed": True}), \
                    patch("experiments.ConceptCheck.validate_stability.checkpoint_fingerprints",
                          side_effect=[first, different, different]) as refresh:
                audited = audit_validation_report(directory)
            self.assertEqual(refresh.call_count, 3)
            self.assertFalse(audited["passed"])
            self.assertFalse(audited["calibrated_health"]["passed"])
            self.assertFalse(audited["repeatability"]["passed"])
            self.assertEqual(audited["repeatability"]["different_components"], ["health_state"])
            self.assertFalse(audited["pacing_invariance"]["passed"])


if __name__ == "__main__":
    unittest.main()
