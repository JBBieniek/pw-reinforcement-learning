"""Explicit numerical settings for reproducible, local training runs.

Call ``configure_determinism`` before seeding or allocating CUDA tensors. These
settings target repeated runs on the same hardware and library versions; they
do not promise bitwise equality across different GPUs or PyTorch releases.
"""

import os

import torch


_CUBLAS_CONFIGS = {":4096:8", ":16:8"}


def configure_determinism(enabled=True):
    """Configure this process and return the actual, JSON-safe runtime flags.

    Deterministic mode uses a supported cuBLAS workspace configuration, strict
    deterministic algorithms, and FP32 operations without TF32. A conflicting
    workspace value fails before any Torch flags are changed. The disabled
    mode retains the legacy project's cuDNN/TF32 settings explicitly.

    Environment changes affect this process and its children only. If CUDA has
    already been initialized, the returned marker records that the caller did
    not follow the required initialization order; existing cuBLAS handles may
    already have been created with another workspace configuration.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    initialized_before = torch.cuda.is_initialized()
    if enabled:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is not None and workspace not in _CUBLAS_CONFIGS:
            raise ValueError(
                "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                "or :16:8 before CUDA initialization; found " + repr(workspace)
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.use_deterministic_algorithms(enabled, warn_only=False)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = not enabled
    torch.backends.cuda.matmul.allow_tf32 = False

    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_initialized_before_configuration": initialized_before,
    }
