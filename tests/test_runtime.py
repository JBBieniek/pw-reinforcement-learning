"""Runtime settings and an isolated CUDA backward reproducibility check."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import torch

from experiments.ConceptCheck.runtime import configure_determinism


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        self.previous_flags = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
        )
        self.addCleanup(self.restore_runtime)

    def restore_runtime(self):
        enabled, warn, cudnn, benchmark, cudnn_tf32, matmul_tf32 = self.previous_flags
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        torch.backends.cudnn.deterministic = cudnn
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        if self.previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.previous_workspace

    def test_enabled_mode_sets_strict_flags_and_default_workspace(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        flags = configure_determinism()
        self.assertTrue(flags["deterministic_algorithms"])
        self.assertTrue(flags["cudnn_deterministic"])
        self.assertFalse(flags["deterministic_warn_only"])
        self.assertFalse(flags["cudnn_benchmark"])
        self.assertFalse(flags["cudnn_allow_tf32"])
        self.assertFalse(flags["matmul_allow_tf32"])
        self.assertEqual(flags["cublas_workspace_config"], ":4096:8")
        json.dumps(flags)

    def test_supported_existing_workspace_is_preserved(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        self.assertEqual(configure_determinism()["cublas_workspace_config"], ":16:8")

    def test_incompatible_workspace_fails_before_mutating_torch_flags(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "unsupported"
        before = self.previous_flags
        with self.assertRaisesRegex(ValueError, "before CUDA initialization"):
            configure_determinism()
        self.assertEqual((
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
        ), before)

    def test_disabled_mode_is_explicit_and_does_not_create_workspace(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        flags = configure_determinism(False)
        self.assertFalse(flags["deterministic_algorithms"])
        self.assertFalse(flags["cudnn_deterministic"])
        self.assertFalse(flags["cudnn_benchmark"])
        self.assertTrue(flags["cudnn_allow_tf32"])
        self.assertFalse(flags["matmul_allow_tf32"])
        self.assertIsNone(flags["cublas_workspace_config"])

    def test_non_boolean_mode_is_rejected(self):
        with self.assertRaises(TypeError):
            configure_determinism("yes")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_repeated_gradients_match_bit_for_bit_in_fresh_process(self):
        script = textwrap.dedent("""
            import json
            import time
            import torch
            from torch import nn
            from experiments.ConceptCheck.runtime import configure_determinism
            flags = configure_determinism()
            assert not flags['cuda_initialized_before_configuration']
            torch.set_num_threads(1)
            torch.manual_seed(189473017)
            model = nn.Sequential(
                nn.Conv2d(4, 16, 3), nn.LeakyReLU(0.01),
                nn.Conv2d(16, 16, 3), nn.LeakyReLU(0.01),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4),
            ).to('cuda')
            batch = torch.rand(8, 4, 84, 84).to('cuda')
            targets = torch.rand(8, 4).to('cuda')
            baseline = None
            losses = []
            for repeat in range(8):
                model.zero_grad(set_to_none=True)
                loss = nn.functional.smooth_l1_loss(model(batch), targets)
                loss.backward()
                gradients = {name: p.grad.detach().cpu().clone()
                             for name, p in model.named_parameters()}
                assert all(torch.isfinite(value).all() for value in gradients.values())
                if baseline is None:
                    baseline = gradients
                else:
                    assert all(torch.equal(baseline[name], value)
                               for name, value in gradients.items())
                losses.append(loss.item())
                time.sleep(0.01)
            assert len(set(losses)) == 1
            print(json.dumps({'repetitions': 8, 'bitwise_identical': True,
                              'device': torch.cuda.get_device_name(0), 'flags': flags}))
        """)
        environment = dict(os.environ)
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=environment, capture_output=True, text=True, timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["repetitions"], 8)
        self.assertTrue(report["bitwise_identical"])
        self.assertFalse(report["flags"]["cuda_initialized_before_configuration"])


if __name__ == "__main__":
    unittest.main()
