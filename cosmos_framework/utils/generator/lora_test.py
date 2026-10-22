# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy

import pytest
import torch

from cosmos_framework.utils.generator.lora import (
    LoraInjectedLinear,
    apply_lora_trainable_scope,
    inject_lora_pre_fsdp,
    merge_lora_adapters_,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 3, bias=True)
        self.lm_head = torch.nn.Linear(3, 2, bias=False)


def test_vlm_lora_preserves_base_keys_and_reports_trainable_scope() -> None:
    model = _Tiny()
    base_keys = set(model.state_dict())
    inject_lora_pre_fsdp(
        model,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules="q_proj",
        lora_dropout=0.0,
        lora_bias="lora_only",
        lora_modules_to_save="lm_head",
    )

    assert isinstance(model.q_proj, LoraInjectedLinear)
    assert base_keys <= set(model.state_dict())
    assert model.q_proj.weight.requires_grad is False
    assert model.q_proj.bias.requires_grad is True
    assert model.lm_head.weight.requires_grad is True
    assert model.q_proj.lora_A.weight.requires_grad is True
    assert model._cosmos_peft_parameter_summary == {
        "training_mode": "peft",
        "trainable_parameters": 23,
        "total_parameters": 35,
        "frozen_parameters": 12,
        "trainable_parameter_tensors": 4,
        "adapter_module_count": 1,
        "adapter_modules": ["q_proj"],
    }

    model.requires_grad_(True)
    summary = apply_lora_trainable_scope(
        model,
        lora_target_modules="q_proj",
        lora_bias="lora_only",
        lora_modules_to_save="lm_head",
    )
    assert summary == model._cosmos_peft_parameter_summary
    assert model.q_proj.weight.requires_grad is False
    assert model.q_proj.bias.requires_grad is True
    assert model.lm_head.weight.requires_grad is True
    assert model.q_proj.lora_A.weight.requires_grad is True


def test_rslora_merge_matches_unmerged_forward() -> None:
    model = _Tiny()
    inject_lora_pre_fsdp(
        model,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules="q_proj",
        lora_use_rslora=True,
    )
    torch.nn.init.normal_(model.q_proj.lora_A.weight)
    torch.nn.init.normal_(model.q_proj.lora_B.weight)
    values = torch.randn(2, 4)
    expected = model.q_proj(values)
    assert merge_lora_adapters_(model) == 1
    with torch.no_grad():
        model.q_proj.lora_A.weight.zero_()
        model.q_proj.lora_B.weight.zero_()
    torch.testing.assert_close(model.q_proj(values), expected)


@pytest.mark.parametrize("prefix", ["", "attention."])
def test_packed_qkv_aliases_preserve_new_lora_options_and_bias_scope(prefix: str) -> None:
    model = torch.nn.Module()
    attention = torch.nn.Module()
    attention.qkv_proj = torch.nn.Linear(4, 9, bias=True)
    model.attention = attention
    targets = ",".join(f"{prefix}{name}" for name in ("q_proj", "k_proj", "v_proj"))

    inject_lora_pre_fsdp(
        model,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=targets,
        lora_dropout=0.1,
        lora_use_rslora=True,
        lora_precision="float32",
        lora_bias="lora_only",
    )

    adapter = attention.qkv_proj
    assert isinstance(adapter, LoraInjectedLinear)
    assert adapter.lora_dropout.p == 0.1
    assert adapter._lora_use_rslora is True
    assert adapter.lora_A.weight.dtype == torch.float32
    assert adapter.weight.requires_grad is False
    assert adapter.bias.requires_grad is True
    assert model._cosmos_peft_parameter_summary["adapter_module_count"] == 1

    model.requires_grad_(True)
    apply_lora_trainable_scope(model, lora_target_modules=targets, lora_bias="lora_only")
    assert adapter.weight.requires_grad is False
    assert adapter.bias.requires_grad is True


def test_packed_lora_rejects_unknown_targets_before_injection() -> None:
    model = torch.nn.Module()
    model.qkv_proj = torch.nn.Linear(4, 9)
    with pytest.raises(ValueError, match="missing_proj"):
        inject_lora_pre_fsdp(
            model,
            lora_rank=2,
            lora_alpha=4,
            lora_target_modules="q_proj,missing_proj",
        )
    assert not isinstance(model.qkv_proj, LoraInjectedLinear)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("use_rslora", [True, False])
@pytest.mark.parametrize(
    "base_dtype,adapter_dtype",
    [
        (torch.float64, torch.float64),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_lora_scaled_intermediate_matches_reference(training, use_rslora, base_dtype, adapter_dtype):
    """Preserve dropout, dtype and gradients against output-scaled LoRA."""
    torch.manual_seed(42)
    layer = LoraInjectedLinear(
        torch.nn.Linear(8, 12, dtype=base_dtype),
        rank=4,
        alpha=3,
        dropout=0.25,
        use_rslora=use_rslora,
        adapter_dtype=adapter_dtype,
    ).train(training)
    torch.nn.init.normal_(layer.lora_A.weight, std=0.2)
    torch.nn.init.normal_(layer.lora_B.weight, std=0.2)
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 3, 8, dtype=base_dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    torch.manual_seed(123)
    actual = layer(x)
    torch.manual_seed(123)
    base = torch.nn.functional.linear(reference_x, reference.weight, reference.bias)
    adapter_input = reference.lora_dropout(reference_x).to(reference.lora_A.weight.dtype)
    adapter_output = reference.lora_B(reference.lora_A(adapter_input)).to(base.dtype)
    expected = base + reference._lora_scale * adapter_output
    assert actual.dtype == base_dtype
    # Moving scale across a projection/cast changes low-precision rounding.
    if base_dtype != adapter_dtype:
        rtol, atol = 0, 0
    elif torch.bfloat16 in (base_dtype, adapter_dtype):
        rtol, atol = 0.03, 0.015
    elif base_dtype == torch.float64:
        rtol, atol = 1e-10, 1e-12
    else:
        rtol, atol = 2e-5, 2e-6
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=rtol, atol=atol)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=rtol, atol=atol)


def test_lora_training_dropout_is_applied_and_disabled_in_eval():
    layer = LoraInjectedLinear(torch.nn.Linear(4, 3), rank=2, alpha=3, dropout=1.0)
    torch.nn.init.ones_(layer.lora_A.weight)
    torch.nn.init.ones_(layer.lora_B.weight)
    values = torch.ones(2, 4)
    base = torch.nn.functional.linear(values, layer.weight, layer.bias)
    torch.testing.assert_close(layer(values), base)
    layer.eval()
    torch.testing.assert_close(layer(values), base + 12.0)
