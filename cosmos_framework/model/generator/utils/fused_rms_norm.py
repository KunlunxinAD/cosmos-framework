# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional fused RMSNorm, backed by a vendor kernel when one is available.

The eager RMSNorm carried by the vendored Qwen3-VL modules upcasts to fp32,
squares, reduces, rsqrts, downcasts, then multiplies by the weight -- six to
seven memory-bound kernels per call, with an fp32 intermediate that doubles the
traffic. Some backends ship a fused kernel pair for exactly this shape; on the
Kunlun XPU backend ``torch_xmlir.nn.rms_norm.RMSNormFunction`` wraps
``custom_ops.rms_layer_norm`` / ``rms_layer_norm_backward``.

Measured on one P800 (hidden 4096, 16384 rows, bfloat16, forward+backward):

* 5.539 ms eager -> 0.918 ms fused (6.0x).
* Closer to a float64 reference than the eager path, not merely equal --
  relative error y 2.92e-3 vs 3.36e-3, dweight 2.88e-3 vs 3.75e-3. The eager
  version downcasts to the input dtype *before* multiplying by the weight and
  loses a rounding step there; the fused kernel keeps its statistics in fp32
  (its ``rstd`` buffer is fp32) and applies the weight inside the kernel.

:func:`maybe_fused_rms_norm` returns ``None`` whenever the fused path is not
usable, so every caller keeps its eager implementation as the fallback and
platforms without the kernel are untouched. ``COSMOS_FUSED_RMSNORM=0`` forces
the eager path -- that is how the baseline arm of a numerical A/B is produced
(see examples/launch_sft_action_policy_droid_p800_precision_ab.sh).
"""

import os

import torch

from cosmos_framework.utils import log

# Resolved on first use, not at import: the probe touches ``torch.ops``, and the
# env override is read once so a run cannot flip implementations mid-flight.
_FUSED_FN = None
_PROBED = False
# One-time reporting: a silent fallback (e.g. a DTensor that was never unsharded)
# would look exactly like "the fusion gave us nothing", so say which path was taken.
_REPORTED_FUSED = False
_REPORTED_FALLBACK = False


def _probe() -> None:
    global _FUSED_FN, _PROBED
    if _PROBED:
        return
    _PROBED = True
    if os.environ.get("COSMOS_FUSED_RMSNORM", "0").strip().lower() in {"0", "false", "no"}:
        return
    try:
        from torch_xmlir.nn.rms_norm import RMSNormFunction  # noqa: PLC0415 - optional backend dep
    except Exception:  # noqa: BLE001 - any import failure means "not available here"
        return
    if not hasattr(torch.ops, "custom_ops") or not hasattr(torch.ops.custom_ops, "rms_layer_norm"):
        return
    _FUSED_FN = RMSNormFunction


def fused_rms_norm_available() -> bool:
    """Whether :func:`maybe_fused_rms_norm` can ever return a tensor in this process."""
    _probe()
    return _FUSED_FN is not None


def maybe_fused_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Fused ``weight * hidden_states / rms(hidden_states)``, or ``None`` to fall back.

    Args:
        hidden_states: Input activations; normalized over the last dimension.
        weight: 1-D gain of length ``hidden_states.shape[-1]``.
        eps: Added to the mean square before the reciprocal square root.

    Returns:
        The normalized tensor, or ``None`` when the fused kernel is unavailable
        or the inputs fall outside what it accepts -- in which case the caller
        must run its own eager implementation.
    """
    _probe()
    fn = _FUSED_FN
    if fn is None:
        return None
    # The kernel allocates its output with ``input.new_empty`` and applies the
    # weight internally, so the two dtypes have to agree; the eager path is the
    # one that tolerates a mismatch (it upcasts). It also indexes raw storage,
    # so a non-contiguous input or a DTensor that has not been unsharded would
    # be wrong or unsupported rather than slow -- and making it contiguous here
    # would reintroduce the copy the fusion exists to remove.
    global _REPORTED_FUSED, _REPORTED_FALLBACK
    if (
        hidden_states.device.type != "cuda"
        or hidden_states.dtype != weight.dtype
        or weight.dim() != 1
        or weight.shape[0] != hidden_states.shape[-1]
        or not hidden_states.is_contiguous()
        or isinstance(hidden_states, torch.distributed.tensor.DTensor)
        or isinstance(weight, torch.distributed.tensor.DTensor)
    ):
        if not _REPORTED_FALLBACK:
            _REPORTED_FALLBACK = True
            log.warning(
                "fused RMSNorm is available but this input falls back to eager: "
                f"device={hidden_states.device.type} x.dtype={hidden_states.dtype} "
                f"w.dtype={weight.dtype} x.contiguous={hidden_states.is_contiguous()} "
                f"x.is_dtensor={isinstance(hidden_states, torch.distributed.tensor.DTensor)} "
                f"w.is_dtensor={isinstance(weight, torch.distributed.tensor.DTensor)} "
                f"w.shape={tuple(weight.shape)} x.shape={tuple(hidden_states.shape)}",
                rank0_only=False,
            )
        return None
    if not _REPORTED_FUSED:
        _REPORTED_FUSED = True
        log.info(f"fused RMSNorm active (custom_ops.rms_layer_norm), first call x.shape={tuple(hidden_states.shape)}")
    return fn.apply(hidden_states, eps, weight, False, False)
