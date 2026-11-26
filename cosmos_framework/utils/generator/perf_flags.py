# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Explicit opt-in flags for experimental Qwen3-VL performance paths."""

import os


def perf_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def get_torch_chunked_ce_tokens() -> int:
    """Zero (or unset) disables the experimental sparse/chunked CE backend."""
    name = "COSMOS_TORCH_CHUNKED_CE_TOKENS"
    raw = os.environ.get(name, "0").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value
