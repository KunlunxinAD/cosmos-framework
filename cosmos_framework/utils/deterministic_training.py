# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU RNG for architecture-independent deterministic training inputs and heads."""

from __future__ import annotations

import hashlib
import math

import torch

_BASE_SEED: int | None = None


def set_base_seed(seed: int) -> None:
    """Remember the training seed before the trainer applies a rank offset."""
    global _BASE_SEED
    _BASE_SEED = seed


@torch.no_grad()
def deterministic_trunc_normal_(tensor: torch.Tensor, std: float, *, name: str) -> torch.Tensor:
    """Initialize the same logical tensor before selecting each DTensor shard.

    The deterministic path uses a private CPU generator, so unrelated draws,
    device RNG implementations, and shard layout do not change skipped heads.
    Ordinary training and meta construction retain the existing initializer.
    """
    if not torch.are_deterministic_algorithms_enabled() or _BASE_SEED is None or tensor.device.type == "meta":
        return torch.nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)

    name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed((_BASE_SEED + name_seed) % (2**63))
    # CPU FP32 erfinv also varies by vector implementation/CPU. Generate in
    # FP64, then round on CPU once before copying the logical values to devices.
    values = torch.empty(tuple(tensor.shape), dtype=torch.float64, device="cpu")
    torch.nn.init.trunc_normal_(values, std=std, a=-3 * std, b=3 * std, generator=generator)
    if hasattr(tensor, "device_mesh"):
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        local_shape, offset = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
        values = values[tuple(slice(start, start + size) for start, size in zip(offset, local_shape, strict=True))]
        local_tensor = tensor.to_local()
    else:
        local_tensor = tensor
    local_tensor.copy_(values.to(dtype=local_tensor.dtype).to(device=local_tensor.device))
    return tensor


def randn_with_generator(size: torch.Size, *, generator: torch.Generator | None, **kwargs) -> torch.Tensor:
    """Use a CPU generator without changing the requested tensor device/dtype."""
    if generator is not None and generator.device.type == "cpu":
        cpu_kwargs = {**kwargs, "device": "cpu"}
        values = torch.randn(size, generator=generator, **cpu_kwargs)
        return values.to(device=kwargs.get("device", "cpu"))
    return torch.randn(size, generator=generator, **kwargs)


def canonical_rotary_frequencies(base: float, dim: int, device=None) -> torch.Tensor:
    """Build non-persistent RoPE frequencies before the single FP32 rounding."""
    exponent = torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim
    values = (1.0 / (base**exponent)).float()
    return values.to(device=device) if device is not None else values


def canonical_timestep_frequencies(dim: int, max_period: int, device=None) -> torch.Tensor:
    """Use identical CPU FP64 frequency construction on every device backend."""
    half = dim // 2
    values = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float64, device="cpu") / half).float()
    return values.to(device=device) if device is not None else values
