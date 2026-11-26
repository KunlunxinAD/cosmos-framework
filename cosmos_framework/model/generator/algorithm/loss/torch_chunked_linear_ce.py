# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in PyTorch-only sparse/chunked LM-head cross entropy.

Batches are reduced in one autograd node: this avoids keeping one large
LM-head weight-gradient buffer per conversation. No Triton dependency.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class _TorchChunkedLinearCESums(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [B, T, H], shifted already
        weight: torch.Tensor,  # [V, H]
        targets: torch.Tensor,  # [B, T], shifted already
        chunk_tokens: int,
        ignore_index: int,
    ) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or weight.ndim != 2
            or targets.ndim != 2
            or hidden.shape[:2] != targets.shape
            or hidden.shape[2] != weight.shape[1]
        ):
            raise ValueError("Expected aligned hidden [B,T,H], weight [V,H], labels [B,T]")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be >0")
        ctx.chunk_tokens = chunk_tokens
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(hidden, weight, targets)
        batch, tokens, width = hidden.shape
        with torch.no_grad():
            selected = torch.nonzero(targets.reshape(-1) != ignore_index, as_tuple=True)[0]
            x = hidden.reshape(-1, width).index_select(0, selected)
            y = targets.reshape(-1).index_select(0, selected)
            sample_ids = selected // tokens
            per_sample = torch.zeros(batch, device=hidden.device, dtype=torch.float32)
            for start in range(0, y.numel(), chunk_tokens):
                end = start + chunk_tokens
                logits = F.linear(x[start:end], weight).float()
                loss = torch.logsumexp(logits, dim=-1) - logits.gather(1, y[start:end, None]).squeeze(1)
                per_sample.index_add_(0, sample_ids[start:end], loss)
        return per_sample

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, targets = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[:2]
        batch, tokens, width = hidden.shape
        dx = torch.zeros_like(hidden) if need_x else None
        # One FP32 accumulation buffer for the entire batch (not per sample).
        dw = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if need_w else None
        with torch.no_grad():
            selected = torch.nonzero(targets.reshape(-1) != ctx.ignore_index, as_tuple=True)[0]
            x = hidden.reshape(-1, width).index_select(0, selected)
            y = targets.reshape(-1).index_select(0, selected)
            sample_ids = selected // tokens
            selected_dx = torch.empty_like(x) if need_x else None
            for start in range(0, y.numel(), ctx.chunk_tokens):
                end = start + ctx.chunk_tokens
                x_chunk = x[start:end]
                y_chunk = y[start:end]
                logits = F.linear(x_chunk, weight).float()
                probs = F.softmax(logits, dim=-1)
                probs.scatter_add_(
                    1,
                    y_chunk[:, None],
                    -torch.ones((y_chunk.numel(), 1), dtype=probs.dtype, device=probs.device),
                )
                # Weight *per-token* gradients before summing because weighted CE
                # can give each conversation a different upstream coefficient.
                probs.mul_(grad_output.index_select(0, sample_ids[start:end]).unsqueeze(1))
                dlogits = probs.to(x_chunk.dtype)
                if need_x:
                    selected_dx[start:end] = dlogits @ weight
                if need_w:
                    dw.add_((dlogits.T @ x_chunk).float())
            if need_x:
                dx.reshape(-1, width).index_copy_(0, selected, selected_dx)
            if need_w:
                dw = dw.to(weight.dtype)
        return dx, dw, None, None, None


def torch_chunked_linear_ce_sums(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    chunk_tokens: int = 512,
) -> torch.Tensor:
    """Return [B] unnormalized CE sums using only valid labels and chunked LM head."""
    return _TorchChunkedLinearCESums.apply(hidden, weight, labels, chunk_tokens, ignore_index)


def torch_chunked_linear_ce_sum(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    chunk_tokens: int = 512,
) -> torch.Tensor:
    """Compatibility entry for one shifted conversation."""
    return torch_chunked_linear_ce_sums(
        hidden.unsqueeze(0), weight, labels.unsqueeze(0), ignore_index=ignore_index, chunk_tokens=chunk_tokens
    ).squeeze(0)
