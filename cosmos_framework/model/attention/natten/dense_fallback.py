# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Attention Subpackage:
Unified implementation for all Attention implementations.

NATTEN Backend: exact dense fallback for devices that cannot run NATTEN.

NATTEN is CUDA-only, so on a Kunlun P800 the ``natten`` entries of ``BACKEND_MAP`` and
``MULTI_DIM_BACKEND_MAP`` resolve to the raising stubs and the whole frame-causal MoT
pathway is closed. The generator asks that slot for exactly two problems, and both are
expressible as *dense* FlashAttention-2 calls, so this module serves those two and raises
for everything else rather than approximating it:

* ``natten_multi_dim_attention`` over a 2-D token layout with a full window and
  ``is_causal=(True, False)``: query frame t attends to every spatial token of frames
  0..t. Decomposed into the per-frame diagonal block plus the binary-split off-diagonal
  rectangles, each one dense Flash2 call, recombined by LSE merge. Exact, not an
  approximation -- every (query, key) frame pair is covered exactly once.
* ``natten_attention`` on a single packed sequence: one dense Flash2 call over the real
  prefix, zero/-inf padded back to the packed length. Kunlun's varlen LSE normalization
  gathers into different storage, which breaks the downstream merge backward's patch into
  the native LSE, so the varlen kernel is avoided here even though flash2 has one.

Genuine neighborhood windows, dilation, stride, and multi-sequence packs raise: they need
a real sparse kernel, and silently widening the mask would train a different model.

Only bound into the natten slot when ``COSMOS_NATTEN_DENSE_FALLBACK=1`` and NATTEN is
unavailable; see ``natten/__init__.py``. Every other environment keeps the stubs.
"""

from __future__ import annotations

import torch
from torch import Tensor

from cosmos_framework.model.attention.flash2 import flash2_attention
from cosmos_framework.model.attention.flash2.checks import flash2_attention_check
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils import log_or_raise_error
from cosmos_framework.model.attention.utils.environment import is_torch_compiling

_UNSUPPORTED = (
    "The dense NATTEN fallback only serves problems that one dense FlashAttention-2 call "
    "can express exactly, and this one is not among them: {reason}. Install NATTEN on a "
    "CUDA device, or choose a backend that implements this case."
)


def _as_tuple(value: tuple | list | int | bool, num_dims: int) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * num_dims


def _mot_merge_helpers():
    """Import the MoT merge helpers on first use.

    ``model.generator.mot`` imports the attention package, so importing these at module
    scope would close an import cycle through this module's own package.
    """
    from cosmos_framework.model.generator.mot.merge_attention import merge_attentions_ac_safe
    from cosmos_framework.model.generator.mot.merge_bridge import MergeAttentionsBridge

    return MergeAttentionsBridge, merge_attentions_ac_safe


def _frame_causal_flash2(
    query: Tensor,  # [B,T,S,H,D]
    key: Tensor,  # [B,T,S,H_KV,D]
    value: Tensor,  # [B,T,S,H_KV,D_V]
    scale: float,
    deterministic: bool,
) -> tuple[Tensor, Tensor]:  # [B,T,S,H,D_V], [B,T,S,H]
    """Q at frame t sees every spatial token in frames 0..t."""
    b, t, s, h, d = query.shape
    kvh = key.shape[-2]
    if t < 1 or key.shape != (b, t, s, kvh, d) or value.shape != key.shape:
        raise ValueError("Expected aligned [B,T,S,H,D] query and key/value tensors.")
    bridge, merge_attentions = _mot_merge_helpers()
    parts = []

    def add(qidx: Tensor, kidx: Tensor, groups: int) -> None:
        nq, nk = qidx.numel() // groups, kidx.numel() // groups
        q_block = query.index_select(1, qidx).reshape(b * groups, nq * s, h, d)
        k_block = key.index_select(1, kidx).reshape(b * groups, nk * s, kvh, d)
        v_block = value.index_select(1, kidx).reshape(b * groups, nk * s, kvh, value.shape[-1])
        out, lse = flash2_attention(
            query=q_block,
            key=k_block,
            value=v_block,
            is_causal=False,
            causal_type=CausalType.DontCare,
            scale=scale,
            return_lse=True,
            deterministic=deterministic,
        )
        lse_shape = lse.shape

        def scatter(o: Tensor, l: Tensor) -> tuple[Tensor, Tensor]:
            full_o = o.new_zeros(b, t, s, h, o.shape[-1])
            full_l = l.new_full((b, t, s, h), torch.finfo(l.dtype).min)
            full_o.index_copy_(1, qidx, o.reshape(b, -1, s, h, o.shape[-1]))
            full_l.index_copy_(1, qidx, l.reshape(b, -1, s, h))
            return full_o.reshape(b, t * s, h, o.shape[-1]), full_l.reshape(b, t * s, h)

        def gather(o: Tensor, l: Tensor) -> tuple[Tensor, Tensor]:
            return (
                o.reshape(b, t, s, h, o.shape[-1]).index_select(1, qidx).reshape(b * groups, nq * s, h, o.shape[-1]),
                l.reshape(b, t, s, h).index_select(1, qidx).reshape(lse_shape),
            )

        parts.append(bridge.apply(out, lse, scatter, gather))

    frames = torch.arange(t, device=query.device)
    add(frames, frames, t)
    # For two distinct frames with key < query, the first binary split that
    # separates them puts the key in a left half and the query in its right half.
    # Every such rectangle is dense; no edge is skipped or counted twice.
    width = 1
    while width < t:
        complete, tail = divmod(t, 2 * width)
        if complete:
            pairs = frames[: complete * 2 * width].reshape(complete, 2, width)
            add(pairs[:, 1].reshape(-1), pairs[:, 0].reshape(-1), complete)
        if tail > width:
            start = complete * 2 * width
            add(frames[start + width :], frames[start : start + width], 1)
        width *= 2

    out, lse = (
        parts[0]
        if len(parts) == 1
        else merge_attentions(outputs=[x[0] for x in parts], lse_tensors=[x[1] for x in parts])
    )
    return out.reshape(b, t, s, h, out.shape[-1]), lse.reshape(b, t, s, h)


def natten_multi_dim_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    window_size: tuple | int = -1,
    stride: tuple | int = 1,
    dilation: tuple | int = 1,
    is_causal: tuple | bool = False,
    scale: float | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Dense stand-in for NATTEN's Multi-Dimensional Attention, frame-causal case only.

    Serves a 2-D token layout whose window covers the whole layout and whose causality is
    causal over frames and bidirectional within a frame. Any other mask is sparse and
    raises.
    """
    if query.ndim != 5:
        raise RuntimeError(_UNSUPPORTED.format(reason=f"the token layout is not 2-D ({query.ndim - 3} dims)"))
    token_layout = tuple(query.shape[1:3])
    window = _as_tuple(window_size, 2)
    # -1 means "cover this dimension"; the frontend normalizes it, direct callers may not.
    window = tuple(token_layout[i] if window[i] == -1 else window[i] for i in range(2))
    if window != token_layout:
        raise RuntimeError(_UNSUPPORTED.format(reason=f"{window=} is a real neighborhood over {token_layout=}"))
    if _as_tuple(stride, 2) != (1, 1) or _as_tuple(dilation, 2) != (1, 1):
        raise RuntimeError(_UNSUPPORTED.format(reason=f"{stride=} and {dilation=} must both be 1 per dimension"))
    if _as_tuple(is_causal, 2) != (True, False):
        raise RuntimeError(_UNSUPPORTED.format(reason=f"{is_causal=} is not frame-causal, spatially bidirectional"))
    if backend_kwargs:
        raise RuntimeError(_UNSUPPORTED.format(reason=f"{backend_kwargs=} are NATTEN kernel options"))

    scale = scale if scale is not None else query.shape[-1] ** -0.5
    output, lse = _frame_causal_flash2(query, key, value, scale=scale, deterministic=deterministic)
    if return_lse:
        return output, lse
    return output


def natten_multi_dim_attention_varlen(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: dict,
    scale: float | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    raise RuntimeError(
        _UNSUPPORTED.format(reason="variable-length multi-dimensional attention packs token layouts per sample")
    )


def natten_attention(
    query: Tensor,  # [B,S_Q,H,D]
    key: Tensor,  # [B,S_KV,H_KV,D]
    value: Tensor,  # [B,S_KV,H_KV,D_V]
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
    """Dense stand-in for NATTEN's plain attention.

    Without varlen offsets this is one dense Flash2 call. With them it serves a single
    packed sequence by slicing to the real prefix and padding the result back, which keeps
    the merge backward's storage patch pointing at the kernel's own saved tensors.
    """
    if backend_kwargs:
        raise RuntimeError(_UNSUPPORTED.format(reason=f"{backend_kwargs=} are NATTEN kernel options"))
    scale = scale if scale is not None else query.shape[-1] ** -0.5
    dense_kwargs = dict(
        is_causal=is_causal,
        causal_type=causal_type,
        scale=scale,
        return_lse=return_lse,
        deterministic=deterministic,
    )
    if cumulative_seqlen_Q is None and cumulative_seqlen_KV is None:
        return flash2_attention(query=query, key=key, value=value, **dense_kwargs)
    if cumulative_seqlen_Q is None or cumulative_seqlen_KV is None:
        raise RuntimeError(_UNSUPPORTED.format(reason="only one side of the varlen offsets was given"))
    if query.shape[0] != 1 or cumulative_seqlen_Q.numel() != 2 or cumulative_seqlen_KV.numel() != 2:
        raise RuntimeError(
            _UNSUPPORTED.format(
                reason=f"a pack of {cumulative_seqlen_Q.numel() - 1} sequences at batch {query.shape[0]} "
                "needs a varlen kernel, and this device's varlen LSE breaks the merge backward"
            )
        )

    q_start, q_end = cumulative_seqlen_Q.tolist()
    k_start, k_end = cumulative_seqlen_KV.tolist()
    if is_causal and causal_type == CausalType.TopLeft:
        # Text self-attention returns zero for the padded query tail. Slice both inputs to
        # the real text prefix; equal Q and K lengths make top-left and bottom-right
        # causality identical.
        if query.shape[1] != key.shape[1] or q_end < k_end:
            raise RuntimeError(
                _UNSUPPORTED.format(reason="top-left causality over unequal real Q and KV lengths keeps its own mask")
            )
        q_end = k_end
        dense_kwargs["causal_type"] = CausalType.BottomRight
    if q_start != 0 or k_start != 0 or not (0 < q_end <= query.shape[1] and 0 < k_end <= key.shape[1]):
        raise RuntimeError(
            _UNSUPPORTED.format(reason=f"the pack's single sequence is not a prefix ({q_start=}, {k_start=})")
        )

    result = flash2_attention(
        query=query[:, :q_end],
        key=key[:, :k_end],
        value=value[:, :k_end],
        **dense_kwargs,
    )
    pad = query.shape[1] - q_end
    if not pad:
        return result

    def pad_output(o: Tensor) -> Tensor:
        return torch.cat((o, o.new_zeros(o.shape[0], pad, *o.shape[2:])), dim=1)

    if not return_lse:
        return pad_output(result)

    def pad_pair(o: Tensor, l: Tensor) -> tuple[Tensor, Tensor]:
        l_pad = l.new_full((l.shape[0], pad, *l.shape[2:]), torch.finfo(l.dtype).min)
        return pad_output(o), torch.cat((l, l_pad), dim=1)

    def take_real(o: Tensor, l: Tensor) -> tuple[Tensor, Tensor]:
        return o[:, :q_end], l[:, :q_end]

    bridge, _ = _mot_merge_helpers()
    return bridge.apply(*result, pad_pair, take_real)


def dense_fallback_attention_check(
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
    return_lse: bool = False,
    is_compiling: bool = False,
    raise_error: bool = False,
) -> bool:
    """Whether the dense fallback can stand in for NATTEN attention on these tensors.

    Every case it serves ends in one dense flash2 call over the same tensors, so flash2's
    own checks are the binding ones. A varlen pack is served by slicing to its single
    sequence, which is why ``is_varlen`` is not forwarded; a pack this fallback cannot
    reduce that way raises from the call itself, with the reason.
    """
    # Equal real Q and KV lengths make top-left and bottom-right causality identical, and
    # flash2 implements only the latter, so that is what the sliced call asks it for. The
    # call verifies the lengths are in fact equal and raises when they are not.
    if is_varlen and is_causal and causal_type == CausalType.TopLeft:
        causal_type = CausalType.BottomRight
    return flash2_attention_check(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=False,
        deterministic=deterministic,
        return_lse=return_lse,
        is_compiling=is_compiling,
        raise_error=raise_error,
    )


def dense_fallback_multi_dim_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """Whether the dense fallback can stand in for NATTEN Multi-Dimensional Attention.

    Only the shapes are known here -- window, stride, dilation and causality reach the
    call itself, which raises for any mask a dense decomposition cannot express exactly.
    A 2-D token layout becomes one dense flash2 problem per rectangle, all of them with
    the folded ``[B, T*S, H, D]`` shape this checks.
    """
    if len(query_shape) != 5:
        log_or_raise_error(
            f"The dense NATTEN fallback only serves a 2-D token layout, got {tuple(query_shape)}.",
            raise_error=raise_error,
        )
        return False
    tokens = query_shape[1] * query_shape[2]
    return flash2_attention_check(
        query_shape=torch.Size((query_shape[0], tokens, query_shape[-2], query_shape[-1])),
        key_shape=torch.Size((key_shape[0], tokens, key_shape[-2], key_shape[-1])),
        value_shape=torch.Size((value_shape[0], tokens, value_shape[-2], value_shape[-1])),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        is_causal=False,
        causal_type=CausalType.DontCare,
        is_varlen=False,
        deterministic=deterministic,
        return_lse=True,
        is_compiling=is_torch_compiling(),
        raise_error=raise_error,
    )
