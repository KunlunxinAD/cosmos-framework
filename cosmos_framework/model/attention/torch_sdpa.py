# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Torch SDPA fallback attention backend.

Wraps `torch.nn.functional.scaled_dot_product_attention` so non-CUDA-flash
hardware (e.g. Baidu Kunlun P800 XPU) can run the cosmos attention frontend
without needing flash_attn / flash3 / natten kernels.

Layout contract matches `BACKEND_MAP` peers: heads-last
(`[B, S, H, D]`). Varlen sequence-packed inputs (`B == 1`,
`cumulative_seqlen_Q/KV` provided) are split per-segment, run through SDPA
individually, then concatenated back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.masks import CausalType


def torch_sdpa_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    # SDPA accepts any reasonable shape on any device. We do not gate on dtype
    # since torch will dispatch a working kernel (math/efficient) for fp16/bf16/fp32.
    if query_shape[-1] != key_shape[-1]:
        if raise_error:
            raise RuntimeError(
                f"torch_sdpa requires Q and K to share head_dim, got {query_shape[-1]=} vs {key_shape[-1]=}."
            )
        return False
    return True


def _sdpa_one(
    q: Tensor,  # [B, S_Q, H, D]
    k: Tensor,  # [B, S_K, H_KV, D]
    v: Tensor,  # [B, S_K, H_KV, D_V]
    is_causal: bool,
    scale: float,
) -> tuple[Tensor, Tensor]:
    # [B,S,H,D] -> [B,H,S,D]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    # GQA: repeat KV heads to match Q heads (SDPA on older torch versions
    # doesn't accept the enable_gqa kwarg).
    h_q = q_t.shape[1]
    h_kv = k_t.shape[1]
    if h_kv != h_q:
        assert h_q % h_kv == 0, f"H_Q ({h_q}) must be divisible by H_KV ({h_kv}) for GQA."
        repeat = h_q // h_kv
        k_t = k_t.repeat_interleave(repeat, dim=1)
        v_t = v_t.repeat_interleave(repeat, dim=1)

    out = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )  # [B, H, S_Q, D_V]
    out = out.transpose(1, 2).contiguous()  # [B, S_Q, H, D_V]
    # Placeholder lse — real merge_attentions path is not used on this backend.
    lse = torch.zeros(
        out.shape[0], out.shape[1], out.shape[2], device=out.device, dtype=torch.float32
    )  # [B, S_Q, H]
    return out, lse


def torch_sdpa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    assert_universal_tensor_checks(query, key, value)
    scale = scale if scale is not None else query.shape[-1] ** -0.5
    is_varlen = cumulative_seqlen_Q is not None

    if not is_varlen:
        out, lse = _sdpa_one(query, key, value, is_causal=is_causal, scale=scale)
        if return_lse:
            return out, lse
        return out

    # Varlen, sequence-packed: B == 1, split along seq dim by cumulative offsets.
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1
    cu_q = cumulative_seqlen_Q.tolist()
    cu_k = cumulative_seqlen_KV.tolist()
    assert len(cu_q) == len(cu_k), f"cu_q and cu_k length mismatch: {len(cu_q)} vs {len(cu_k)}"

    out_chunks: list[Tensor] = []
    lse_chunks: list[Tensor] = []
    for i in range(len(cu_q) - 1):
        q_s, q_e = cu_q[i], cu_q[i + 1]
        k_s, k_e = cu_k[i], cu_k[i + 1]
        if q_e == q_s:
            continue
        q_i = query[:, q_s:q_e]  # [1, sq, H, D]
        k_i = key[:, k_s:k_e]
        v_i = value[:, k_s:k_e]
        out_i, lse_i = _sdpa_one(q_i, k_i, v_i, is_causal=is_causal, scale=scale)
        out_chunks.append(out_i)
        lse_chunks.append(lse_i)

    out = torch.cat(out_chunks, dim=1)  # [1, total_q_tokens, H, D_V]
    if return_lse:
        lse = torch.cat(lse_chunks, dim=1)  # [1, total_q_tokens, H]
        return out, lse
    return out
