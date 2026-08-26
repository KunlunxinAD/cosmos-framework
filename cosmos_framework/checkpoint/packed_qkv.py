# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Utilities for converting unpacked QKV checkpoint tensors to packed QKV."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

import torch
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, TensorStorageMetadata


_QKV_KEY_PATTERN = re.compile(
    r"^(?P<prefix>.*\.self_attn\.)(?P<component>[qkv])_proj"
    r"(?P<pathway>_moe_gen)?\.(?P<value_kind>weight|bias)$"
)
_QKV_COMPONENTS = ("q", "k", "v")


@dataclass(frozen=True)
class PackedQKVConversionReport:
    source_tensor_count: int
    output_tensor_count: int
    packed_tensor_count: int
    source_qkv_tensor_count: int
    packed_keys: tuple[str, ...]


@dataclass(frozen=True)
class PackedGateUpConversionReport:
    source_tensor_count: int
    output_tensor_count: int
    packed_tensor_count: int
    source_gate_up_tensor_count: int
    packed_keys: tuple[str, ...]


def _packed_key_from_match(match: re.Match[str]) -> str:
    pathway = match.group("pathway") or ""
    return f'{match.group("prefix")}qkv_proj{pathway}.{match.group("value_kind")}'


def pack_qkv_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], PackedQKVConversionReport]:
    """Replace every complete q/k/v projection group with one dim-0 concatenation."""
    groups: dict[str, dict[str, torch.Tensor]] = {}
    source_to_target: dict[str, str] = {}

    for key, tensor in state_dict.items():
        match = _QKV_KEY_PATTERN.fullmatch(key)
        if match is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"QKV checkpoint value {key!r} is not a tensor: {type(tensor).__name__}")
        target_key = _packed_key_from_match(match)
        component = match.group("component")
        group = groups.setdefault(target_key, {})
        if component in group:
            raise ValueError(f"Duplicate {component.upper()} tensor for packed target {target_key}")
        group[component] = tensor
        source_to_target[key] = target_key

    if not groups:
        raise ValueError("No unpacked self-attention Q/K/V projection groups found in checkpoint")

    incomplete = {
        target_key: sorted(set(_QKV_COMPONENTS) - set(parts))
        for target_key, parts in groups.items()
        if set(parts) != set(_QKV_COMPONENTS)
    }
    if incomplete:
        details = "; ".join(f"{key} missing {missing}" for key, missing in sorted(incomplete.items()))
        raise ValueError(f"Incomplete QKV projection groups: {details}")

    packed_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    emitted_targets: set[str] = set()
    for key, tensor in state_dict.items():
        target_key = source_to_target.get(key)
        if target_key is None:
            packed_state[key] = tensor
            continue
        if target_key in emitted_targets:
            continue

        parts = groups[target_key]
        dtypes = {parts[name].dtype for name in _QKV_COMPONENTS}
        devices = {parts[name].device for name in _QKV_COMPONENTS}
        trailing_shapes = {tuple(parts[name].shape[1:]) for name in _QKV_COMPONENTS}
        if len(dtypes) != 1 or len(devices) != 1 or len(trailing_shapes) != 1:
            raise ValueError(
                f"Incompatible Q/K/V tensors for {target_key}: "
                f"dtypes={dtypes}, devices={devices}, trailing_shapes={trailing_shapes}"
            )
        packed_state[target_key] = torch.cat([parts[name] for name in _QKV_COMPONENTS], dim=0)
        emitted_targets.add(target_key)

    report = PackedQKVConversionReport(
        source_tensor_count=len(state_dict),
        output_tensor_count=len(packed_state),
        packed_tensor_count=len(groups),
        source_qkv_tensor_count=len(source_to_target),
        packed_keys=tuple(sorted(groups)),
    )
    return packed_state, report


_GATE_UP_KEY_PATTERN = re.compile(
    r"^(?P<prefix>.*\.(?:mlp|mlp_moe_gen)\.)(?P<component>gate|up)_proj\.(?P<value_kind>weight|bias)$"
)
_GATE_UP_COMPONENTS = ("gate", "up")


def pack_gate_up_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], PackedGateUpConversionReport]:
    """Replace every dense MoT MLP gate/up pair with one dim-0 concatenation."""
    groups: dict[str, dict[str, torch.Tensor]] = {}
    source_to_target: dict[str, str] = {}

    for key, tensor in state_dict.items():
        match = _GATE_UP_KEY_PATTERN.fullmatch(key)
        if match is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Gate/Up checkpoint value {key!r} is not a tensor: {type(tensor).__name__}")
        target_key = f'{match.group("prefix")}gate_up_proj.{match.group("value_kind")}'
        component = match.group("component")
        group = groups.setdefault(target_key, {})
        if component in group:
            raise ValueError(f"Duplicate {component} tensor for packed target {target_key}")
        group[component] = tensor
        source_to_target[key] = target_key

    if not groups:
        raise ValueError("No unpacked dense MLP gate/up projection groups found in checkpoint")

    incomplete = {
        target_key: sorted(set(_GATE_UP_COMPONENTS) - set(parts))
        for target_key, parts in groups.items()
        if set(parts) != set(_GATE_UP_COMPONENTS)
    }
    if incomplete:
        details = "; ".join(f"{key} missing {missing}" for key, missing in sorted(incomplete.items()))
        raise ValueError(f"Incomplete Gate/Up projection groups: {details}")

    packed_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    emitted_targets: set[str] = set()
    for key, tensor in state_dict.items():
        target_key = source_to_target.get(key)
        if target_key is None:
            packed_state[key] = tensor
            continue
        if target_key in emitted_targets:
            continue

        parts = groups[target_key]
        dtypes = {parts[name].dtype for name in _GATE_UP_COMPONENTS}
        devices = {parts[name].device for name in _GATE_UP_COMPONENTS}
        trailing_shapes = {tuple(parts[name].shape[1:]) for name in _GATE_UP_COMPONENTS}
        if len(dtypes) != 1 or len(devices) != 1 or len(trailing_shapes) != 1:
            raise ValueError(
                f"Incompatible Gate/Up tensors for {target_key}: "
                f"dtypes={dtypes}, devices={devices}, trailing_shapes={trailing_shapes}"
            )
        packed_state[target_key] = torch.cat([parts[name] for name in _GATE_UP_COMPONENTS], dim=0)
        emitted_targets.add(target_key)

    report = PackedGateUpConversionReport(
        source_tensor_count=len(state_dict),
        output_tensor_count=len(packed_state),
        packed_tensor_count=len(groups),
        source_gate_up_tensor_count=len(source_to_target),
        packed_keys=tuple(sorted(groups)),
    )
    return packed_state, report


def allocate_state_dict_from_dcp_metadata(metadata: Metadata) -> OrderedDict[str, torch.Tensor]:
    """Allocate CPU tensors matching a tensor-only DCP model component."""
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    byte_keys: list[str] = []
    for key, storage_metadata in metadata.state_dict_metadata.items():
        if isinstance(storage_metadata, TensorStorageMetadata):
            state_dict[key] = torch.empty(
                tuple(storage_metadata.size),
                dtype=storage_metadata.properties.dtype,
                device="cpu",
            )
        elif isinstance(storage_metadata, BytesStorageMetadata):
            byte_keys.append(key)
        else:
            raise TypeError(f"Unsupported DCP metadata for {key!r}: {type(storage_metadata).__name__}")

    if byte_keys:
        raise ValueError(f"Model-only DCP contains non-tensor entries: {sorted(byte_keys)}")
    return state_dict


def state_dict_nbytes(state_dict: Mapping[str, Any]) -> int:
    return sum(value.numel() * value.element_size() for value in state_dict.values() if isinstance(value, torch.Tensor))
