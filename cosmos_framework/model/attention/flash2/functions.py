# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v2 (flash2) Backend: intermediate APIs
Only safe to import when FLASH2_SUPPORTED is True.
"""

import os

import torch
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from torch import Tensor

from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.flash2.checks import flash2_attention_check
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils.environment import is_torch_compiling


# --- COSMOS_FLASH2_SAC_CUSTOM_OP ---
def _maybe_enable_flash2_sac_custom_op() -> None:
    if os.environ.get("COSMOS_FLASH2_SAC_CUSTOM_OP", "0") != "1":
        return
    import flash_attn.flash_attn_interface as fai

    if getattr(fai, "_cosmos_sac_custom_op_enabled", False):
        return

    orig_fwd = fai._flash_attn_forward
    orig_varlen_fwd = fai._flash_attn_varlen_forward

    @torch.library.custom_op("flash_attn::_flash_attn_forward", mutates_args=())
    def _flash_attn_forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        alibi_slopes: torch.Tensor | None,
        return_softmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _q, _k, _v, _out_padded, softmax_lse, s_dmask, rng_state = orig_fwd(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            (window_size_left, window_size_right),
            alibi_slopes,
            return_softmax,
        )
        return out, softmax_lse, s_dmask, rng_state

    @torch.library.custom_op("flash_attn::_flash_attn_varlen_forward", mutates_args=())
    def _flash_attn_varlen_forward_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        alibi_slopes: torch.Tensor | None,
        return_softmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _q, _k, _v, _out_padded, softmax_lse, s_dmask, rng_state = orig_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            (window_size_left, window_size_right),
            alibi_slopes,
            return_softmax,
        )
        return out, softmax_lse, s_dmask, rng_state

    def _wrapped_flash_attn_forward(
        q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax
    ):
        if q.shape[-1] % 8 != 0:
            return orig_fwd(
                q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax
            )
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        out, softmax_lse, s_dmask, rng_state = torch.ops.flash_attn._flash_attn_forward(
            q,
            k,
            v,
            float(dropout_p),
            float(softmax_scale),
            bool(causal),
            int(window_size[0]),
            int(window_size[1]),
            alibi_slopes,
            bool(return_softmax),
        )
        return out, q, k, v, out, softmax_lse, s_dmask, rng_state

    def _wrapped_flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        return_softmax,
    ):
        if q.shape[-1] % 8 != 0:
            return orig_varlen_fwd(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                return_softmax,
            )
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        out, softmax_lse, s_dmask, rng_state = torch.ops.flash_attn._flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            float(dropout_p),
            float(softmax_scale),
            bool(causal),
            int(window_size[0]),
            int(window_size[1]),
            alibi_slopes,
            bool(return_softmax),
        )
        return out, q, k, v, out, softmax_lse, s_dmask, rng_state

    fai._flash_attn_forward = _wrapped_flash_attn_forward
    fai._flash_attn_varlen_forward = _wrapped_flash_attn_varlen_forward
    fai._cosmos_sac_custom_op_enabled = True


_maybe_enable_flash2_sac_custom_op()


def _unpad_lse_bhm(lse_: Tensor, cumulative_seqlen_Q: Tensor, total_tokens: int) -> Tensor:
    """Un-pad a ``[batch, H, max_seqlen]`` LSE into the packed ``[H, total_tokens]`` layout.

    Builds the gather index on the device. The obvious implementation --
    ``(cu[1:] - cu[:-1]).tolist()`` plus a Python-loop ``torch.cat`` -- costs one
    device->host sync per attention call, which drains the device queue and stops the
    host from running ahead; on P800 that is 640 syncs per rank per training step and it
    prevents FSDP/CP collectives from overlapping with compute. ``total_tokens`` comes
    from a tensor *shape*, so it is a Python int already and needs no sync.
    """
    b, h, m = lse_.shape
    pos = torch.arange(total_tokens, device=lse_.device, dtype=torch.int64)
    cu = cumulative_seqlen_Q.to(torch.int64)
    # Sample owning each packed token = #{boundaries <= pos} over cu[1:]; the upper-bound
    # search also skips zero-length samples, matching the sliced-cat behaviour exactly.
    sample = torch.searchsorted(cu[1:].contiguous(), pos, right=True)
    flat = sample * m + (pos - cu[sample])
    return lse_.permute(1, 0, 2).reshape(h, b * m).index_select(1, flat)


def flash2_attention(
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
    cumulative_seqlen_Q_host: Tensor | None = None,
    cumulative_seqlen_KV_host: Tensor | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs Flash Attention v2 on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        cumulative_seqlen_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (int | None): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (int | None): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

        cumulative_seqlen_Q_host (Tensor | None): (varlen) Optional host-resident duplicate of
            `cumulative_seqlen_Q`. When both host arrays are given they are what gets handed to
            `flash_attn_varlen_func`, which is strictly a host-side saving on this runtime --
            see the note at the call site. Must hold the same values as the device arrays; that
            is the caller's guarantee, not something checked here.

        cumulative_seqlen_KV_host (Tensor | None): (varlen) The same, for `cumulative_seqlen_KV`.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to Flash's
            attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not contiguous in this backend (Flash2) and it should not be made
            contiguous unless we can guarantee its results aren't merged via `merge_attentions`.
    """

    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]
    deterministic = deterministic or torch.are_deterministic_algorithms_enabled()

    assert flash2_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )

    # This check introduces recompiles
    if not is_torch_compiling():
        if is_varlen and max_seqlen_Q == max_seqlen_KV == 0:
            raise NotImplementedError(
                "You're trying to use varlen attention with the flash2 backend and "
                "an empty batch, which is not yet supported by flash2."
            )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if is_varlen:
        assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        q = query.squeeze(0)  # [total_tokens,H,D]
        k = key.squeeze(0)  # [total_tokens,Hkv,D]
        v = value.squeeze(0)  # [total_tokens,Hkv,Dv]
        assert q.dim() == k.dim() == v.dim() == 3
        # Hand the kernel the host ranges when the caller kept a copy. Some builds of this
        # entry point -- the Kunlun XPU one in particular -- read the ranges on the host and
        # copy them back themselves with a *blocking* device->host transfer preceded by a wait
        # on the stream, so every varlen call stalls the host on all device work queued before
        # it. That stall is what stops the host from running ahead and keeps FSDP2's implicit
        # all-gather prefetch from overlapping with compute. Where the entry point instead
        # takes a host array directly it skips both the wait and the copy. Verified bit-equal
        # on P800 (output and all three input gradients) against passing the device arrays,
        # with identical device-side time -- nothing downstream of the copy changes. Builds
        # that keep the ranges on the device upload these instead, which is the same transfer
        # the device arrays already cost and no worse.
        cu_seqlens_q = cumulative_seqlen_Q if cumulative_seqlen_Q_host is None else cumulative_seqlen_Q_host
        cu_seqlens_k = cumulative_seqlen_KV if cumulative_seqlen_KV_host is None else cumulative_seqlen_KV_host
        out, lse_, _ = flash_attn_varlen_func(
            q=query.squeeze(0),
            k=key.squeeze(0),
            v=value.squeeze(0),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_Q,
            max_seqlen_k=max_seqlen_KV,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
            # block_table=None,
        )
        assert out.dim() == 3  # [total_tokens,H,Dv]
        output = out.unsqueeze(0)  # [1,total_tokens,H,Dv]

        # Normalize LSE to [H, total_tokens] only when requested. Some Kunlun builds
        # expose a 3-D buffer even when the kernel writes packed head-major LSE.
        lse = None
        if return_lse:
            if lse_.dim() == 3:
                if os.environ.get("COSMOS_FLASH2_VARLEN_PACKED_LSE") == "1":
                    # Keep a view of the packed prefix so merge backward can write it back.
                    _, heads, _ = lse_.shape
                    total_queries = q.shape[0]
                    lse_ = lse_.view(-1)[: heads * total_queries].view(heads, total_queries)
                else:
                    lse_ = _unpad_lse_bhm(lse_, cumulative_seqlen_Q, q.shape[0])
            assert lse_.dim() == 2  # [H,total_tokens]
            lse = lse_.unsqueeze(0)  # [1,H,total_tokens]

    else:
        output, lse, _ = flash_attn_func(  # output: [B,N,H,Dv], lse: [B,H,N]
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
        )

    assert isinstance(output, Tensor)
    assert output.dim() == 4  # [B,N,H,Dv] or [1,total_tokens,H,Dv]

    if not return_lse:
        return output

    assert isinstance(lse, Tensor)
    assert lse.dim() == 3  # [B,H,N] or [1,H,total_tokens]

    # NOTE: Do NOT call .contiguous on LSE, otherwise Attention Merging backward pass will be
    # incorrect. All output and lse tensors passed into `merge_attentions` must have the same data
    # pointer as their corresponding attention autograd ops!
    # Affected Kunlun dense builds store LSE head-major across batches.
    # Normalize every dense caller's LSE here, including teacher-forcing chunks
    # that bypass the frame-causal fallback.
    if not is_varlen and lse.shape[0] > 1 and os.environ.get(
        "COSMOS_FLASH2_LSE_BHFIX", "0"
    ) not in {"0", "", "false", "no"}:
        batches, heads, tokens = lse.shape
        lse = lse.view(heads, batches, tokens).permute(1, 0, 2)
    lse = lse.permute(0, 2, 1)  # [B,N,H] or [1,total_tokens,H]

    return output, lse
