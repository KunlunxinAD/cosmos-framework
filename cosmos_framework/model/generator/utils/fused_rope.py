# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional fused rotary embedding for the packed (T, H, D) attention layout.

The eager ``apply_rotary_pos_emb`` is six memory-bound ops per call -- two
multiplies and an add for each of q and k, plus the ``rotate_half`` slice /
negate / concat that materializes a second copy of both tensors. Some backends
ship a single kernel for it; on the Kunlun XPU backend that is
``torch_xmlir.nn.rope.FusedRoPEFunc`` over ``custom_ops.fused_rope_forward``.

Measured on one P800 (Hq=32, Hkv=8, D=128, bfloat16, forward+backward over both
q and k): 8.554 -> 4.925 ms at 32k tokens and 16.963 -> 9.854 ms at 64k tokens,
i.e. -42% at both sizes, matching the eager ``rotate_half`` result to 2.84e-03
(bfloat16 rounding level).

The kernel's THD entry point is particular, and every constraint below was
found by hitting its check messages rather than from documentation:

* ``query`` must be 3-D ``[T, H, D]`` -- which is exactly the layout the MoT
  attention already has at this point -- and ``key`` must be ``None``, so q and
  k take one call each.
* ``cos`` / ``sin`` must be 4-D ``[T, 1, 1, D]``. Anything lower-rank is
  rejected with "expected 4D tensor", and a leading batch dim with T in the
  second position with "expect cos seq size to be 1".
* ``cu_seqlens`` must be int64; int32 is rejected with "expected Long type".
* ``interleaved=True`` is BLHD-only, so the half-rotation convention that
  matches ``rotate_half`` is ``interleaved=False``.

:func:`maybe_fused_rope` returns ``None`` whenever any of that does not hold --
notably for the unpacked 4-D ``[B, H, N, D]`` path, which keeps the eager
implementation -- so callers always retain their own fallback and platforms
without the kernel are untouched. ``COSMOS_FUSED_ROPE=0`` forces the eager path,
which is how the baseline arm of a numerical A/B is produced.
"""

import os

import torch

from cosmos_framework.utils import log

_FUSED_FN = None
_PROBED = False
_REPORTED_FUSED = False
_REPORTED_FALLBACK = False
# ``cu_seqlens`` is the same two-element tensor for every call within a step, and
# building it from a Python list is a host-to-device copy: at 72 attention modules
# x 2 tensors that is ~144 tiny transfers per micro-step, all for the same value.
_CU_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _probe() -> None:
    global _FUSED_FN, _PROBED
    if _PROBED:
        return
    _PROBED = True
    if os.environ.get("COSMOS_FUSED_ROPE", "0").strip().lower() in {"0", "false", "no"}:
        return
    try:
        from torch_xmlir.nn.rope import FusedRoPEFunc  # noqa: PLC0415 - optional backend dep
    except Exception:  # noqa: BLE001 - any import failure means "not available here"
        return
    if not hasattr(torch.ops, "custom_ops") or not hasattr(torch.ops.custom_ops, "fused_rope_forward"):
        return
    _FUSED_FN = FusedRoPEFunc


def fused_rope_available() -> bool:
    """Whether :func:`maybe_fused_rope` can ever return tensors in this process."""
    _probe()
    return _FUSED_FN is not None


def _usable(t: torch.Tensor, ndim: int) -> bool:
    return (
        t.device.type == "cuda"
        and t.dim() == ndim
        and t.is_contiguous()
        and not isinstance(t, torch.distributed.tensor.DTensor)
    )


def maybe_fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Rotary embedding for packed q/k, or ``None`` to fall back to eager.

    Args:
        q: Query, ``[T, num_heads, head_dim]``.
        k: Key, ``[T, num_kv_heads, head_dim]``.
        cos: Per-token cosines, ``[T, head_dim]``.
        sin: Per-token sines, ``[T, head_dim]``.

    Returns:
        ``(q_embed, k_embed)`` with the same shapes as ``q`` / ``k``, or ``None``
        when the fused kernel is unavailable or these inputs fall outside what it
        accepts -- in which case the caller must run its own eager path.
    """
    global _REPORTED_FUSED, _REPORTED_FALLBACK
    _probe()
    fn = _FUSED_FN
    if fn is None:
        return None
    ok = (
        _usable(q, 3)
        and _usable(k, 3)
        and _usable(cos, 2)
        and _usable(sin, 2)
        and q.dtype == k.dtype == cos.dtype == sin.dtype
        and cos.shape == sin.shape
        and cos.shape[0] == q.shape[0] == k.shape[0]
        and cos.shape[1] == q.shape[2] == k.shape[2]
        and cos.shape[1] % 2 == 0
    )
    if not ok:
        if not _REPORTED_FALLBACK:
            _REPORTED_FALLBACK = True
            log.warning(
                "fused RoPE is available but these inputs fall back to eager: "
                f"q={tuple(q.shape)}/{q.dtype}/contig={q.is_contiguous()} "
                f"k={tuple(k.shape)}/{k.dtype}/contig={k.is_contiguous()} "
                f"cos={tuple(cos.shape)}/{cos.dtype}/contig={cos.is_contiguous()} "
                f"device={q.device.type}",
                rank0_only=False,
            )
        return None
    # ``unsqueeze`` twice rather than ``reshape``: a view costs nothing, while a
    # reshape of a non-contiguous cos would copy back the traffic this removes.
    cos4 = cos.unsqueeze(1).unsqueeze(1)
    sin4 = sin.unsqueeze(1).unsqueeze(1)
    # One packed sequence: the per-token cos/sin already carry all position
    # information, so the boundaries only have to cover [0, T). Cached per
    # (T, device); the packed length repeats across every layer of a step.
    key = (q.shape[0], q.device)
    cu = _CU_CACHE.get(key)
    if cu is None:
        cu = torch.tensor([0, q.shape[0]], dtype=torch.int64, device=q.device)
        _CU_CACHE[key] = cu
    q_embed, _ = fn.apply(q, None, None, cos4, sin4, cu, "THD", False)
    k_embed, _ = fn.apply(k, None, None, cos4, sin4, cu, "THD", False)
    if not _REPORTED_FUSED:
        _REPORTED_FUSED = True
        log.info(f"fused RoPE active (custom_ops.fused_rope_forward), first call q={tuple(q.shape)}")
    return q_embed, k_embed
