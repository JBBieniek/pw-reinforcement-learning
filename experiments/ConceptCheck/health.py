"""Observe the visual representation without changing training or its RNG."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import torch
from torch import nn


class RepresentationCollapseError(RuntimeError):
    """Repeated diverse batches no longer produce input-dependent features or Q values."""


def _number(value):
    number = float(value)
    return number if math.isfinite(number) else None


@torch.no_grad()
def inspect_network_health(model, observations):
    """Inspect a deterministic CNN batch; never draw RNG or alter module modes.

    The supported network is the project's stateless Conv/Linear CNN. Models
    with dropout or running-statistic layers are deliberately skipped instead of
    disturbing their RNG or buffers during a diagnostic forward pass.
    Existing gradients are read, never cleared or recomputed.
    """
    conv = getattr(model, "conv", None)
    fc = getattr(model, "fc", None)
    if not isinstance(conv, nn.Sequential) or not isinstance(fc, nn.Sequential):
        return {"supported": False, "reason": "model has no supported conv/fc representation"}
    allowed = (nn.Sequential, nn.Conv2d, nn.Linear, nn.ReLU, nn.LeakyReLU, nn.Flatten)
    if any(not isinstance(module, allowed) for branch in (conv, fc) for module in branch.modules()):
        return {"supported": False, "reason": "representation includes stateful or stochastic layers"}
    if not isinstance(observations, torch.Tensor) or observations.ndim < 2 or not len(observations):
        raise ValueError("observations must be a nonempty batched tensor")

    features = conv(observations.float() / 255.0)
    flat_features = features.reshape(features.shape[0], -1)
    q_values = fc(features)
    if q_values.ndim != 2 or q_values.shape[0] != len(observations):
        raise ValueError("model must return [batch, actions] Q values")
    diverse = bool(len(observations) > 1 and torch.any(observations[1:] != observations[:1]).item())
    trainable = [parameter for parameter in conv.parameters() if parameter.requires_grad]
    gradients = [parameter.grad.detach() for parameter in trainable if parameter.grad is not None]
    gradients_present = bool(trainable) and len(gradients) == len(trainable)
    gradients_finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    gradient_norm = math.sqrt(sum(float(gradient.double().square().sum().item()) for gradient in gradients)) if gradients else None
    gradient_zero = all(int(torch.count_nonzero(gradient).item()) == 0 for gradient in gradients) if gradients_present else None
    finite = bool(torch.isfinite(flat_features).all().item() and torch.isfinite(q_values).all().item()
                  and gradients_finite)
    actions = q_values.argmax(dim=1)
    counts = torch.bincount(actions, minlength=q_values.shape[1]).cpu().tolist()
    return {
        "supported": True,
        "diverse_observations": diverse,
        "finite": finite,
        "features_all_zero": int(torch.count_nonzero(flat_features).item()) == 0,
        "feature_std_across_states": _number(flat_features.std(dim=0, unbiased=False).mean().item()),
        "nonzero_fraction": float(torch.count_nonzero(flat_features).item() / flat_features.numel()),
        "feature_min": _number(flat_features.min().item()),
        "feature_max": _number(flat_features.max().item()),
        "q_std_across_states": _number(q_values.std(dim=0, unbiased=False).mean().item()),
        "greedy_action_counts": {str(index): int(count) for index, count in enumerate(counts)},
        "conv_gradient_norm": _number(gradient_norm) if gradient_norm is not None else None,
        "conv_gradients_present": gradients_present,
        "conv_gradients_all_zero": gradient_zero,
    }


class HealthMonitor:
    def __init__(self, path, patience=3):
        if not isinstance(patience, int) or isinstance(patience, bool) or patience <= 0:
            raise ValueError("patience must be a positive integer")
        self.path = Path(path)
        self.history_path = self.path.with_suffix(".jsonl")
        self.patience = patience
        self.consecutive_collapsed_checks = 0
        self.checks = 0

    def get_state(self):
        return {"consecutive_collapsed_checks": self.consecutive_collapsed_checks,
                "checks": self.checks}

    def set_state(self, state):
        restored = {}
        for key in ("consecutive_collapsed_checks", "checks"):
            value = state.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid health monitor state: {key}")
            restored[key] = value
        for key, value in restored.items():
            setattr(self, key, value)

    def check(self, model, observations, step):
        report = inspect_network_health(model, observations)
        reasons = []
        if (report.get("supported") and report["finite"] and report["diverse_observations"]
                and report["conv_gradients_present"]):
            if report["features_all_zero"] and report["conv_gradients_all_zero"]:
                reasons.append("zero_features_and_zero_cnn_gradients")
            if not report["features_all_zero"] and report["feature_std_across_states"] == 0.0:
                reasons.append("constant_nonzero_features")
            if report["q_std_across_states"] == 0.0:
                reasons.append("constant_q_values")
        collapsed = bool(reasons)
        self.consecutive_collapsed_checks = self.consecutive_collapsed_checks + 1 if collapsed else 0
        self.checks += 1
        status = ("unsupported" if not report.get("supported") else
                  "non_finite" if not report["finite"] else
                  "collapsed" if collapsed else "healthy")
        report.update(
            step=int(step), timestamp_utc=datetime.now(timezone.utc).isoformat(),
            checks=self.checks, consecutive_collapsed_checks=self.consecutive_collapsed_checks,
            collapsed=collapsed, collapse_reasons=reasons,
            status=status,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, allow_nan=False) + "\n")
        if report.get("supported") and not report["finite"]:
            raise RepresentationCollapseError(f"Non-finite visual representation or gradients at step {step}")
        if self.consecutive_collapsed_checks >= self.patience:
            raise RepresentationCollapseError(
                f"Visual representation collapsed at step {step}: diverse observations produce "
                f"{', '.join(reasons)} for {self.consecutive_collapsed_checks} checks"
            )
        return report
