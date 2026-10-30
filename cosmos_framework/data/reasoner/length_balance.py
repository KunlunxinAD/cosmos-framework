# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Length-balanced placement of samples onto (rank, microstep) slots.

Why: FSDP all_gather in the VideoPhy-2 P800 recipe is dominated by RANK ARRIVAL
SKEW, not transfer. Measured on the current baseline (2026-10-28, 4040 collective
instances x 8 ranks):

    observed per-rank duration   median 15.800 ms
    arrival skew max-min(start)  median 17.294 ms
    transfer after last arrival  median  5.494 ms
    end-of-collective spread     median  0.068 ms   <- ranks finish together

Ranks finish together but arrive 17 ms apart, so the collective is waiting for the
slowest rank. The skew is data-induced: sample lengths are 976 / 2672 / 5632
tokens (5.8x spread), so whichever rank holds a long sample in a given microstep
arrives late and every other rank stalls.

This module reorders the global index list so that, at each microstep, all 8 ranks
hold samples of similar length. A Monte-Carlo over the measured length
distribution puts the reduction in per-microstep length spread at ~88.7%, with
6.25 of 8 microsteps becoming fully homogeneous.

CORRECTNESS -- why this does not change the optimisation problem:

The reorder is a PERMUTATION WITHIN each block of ``world_size * grad_accum``
indices, and an aligned block is exactly one optimizer step's sample set. So for
every step the set of samples is identical to the unbalanced order; each sample
still appears exactly once; each rank still gets exactly ``grad_accum`` samples
so the GA average keeps every per-sample weight at 1/GA. Only which rank and
which microstep a sample lands on changes, i.e. only the floating-point
accumulation order. That is the same correctness tier as FSDP delayed
reduce-scatter (mathematically equivalent, not bit-exact).

"ALIGNED" IS LOAD-BEARING. Ranks consume their lanes as one continuous stream
across epochs, so the block grid is anchored to the GLOBAL step index, not to
each epoch's index list. When the epoch length is not a multiple of the block,
the grid drifts by ``epoch_len % block`` positions every epoch; blocks then
straddle step boundaries and the permutation can move a sample between adjacent
steps, which DOES change the objective. Callers must therefore pass
``block_offset=epoch_block_offset(epoch, epoch_len, world_size, grad_accum)``.
VideoPhy-2 has ``3343 % 64 == 15``: with the offset omitted, set invariance
broke at 448 of 500 steps and the gain decayed from 22.13 to 26.56 s/step.
The leading and trailing partial blocks are left in stock order, which costs
about one unbalanced step per epoch (1/52 here).

Placement formula, derived from ``IterableDistributor`` (round-robin over
``dp_rank * num_workers + worker_id``) with ``num_workers`` workers per rank and
the DataLoader's round-robin across workers. Rank ``r``'s microstep ``k`` consumes

    L[block + num_workers * world_size * (k // num_workers) + num_workers * r + (k % num_workers)]

For world_size=8, num_workers=2, grad_accum=8 the offsets
``16 * (k // 2) + 2 * r + (k % 2)`` cover 0..63 exactly once, which is what makes
the block equal to one optimizer step.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Sequence


def slot_offset(rank: int, microstep: int, world_size: int, num_workers: int) -> int:
    """Offset inside a block for (rank, microstep) under round-robin sharding."""
    w = max(1, num_workers)
    return w * world_size * (microstep // w) + w * rank + (microstep % w)


def load_length_index(path: str) -> dict[str, int]:
    """Read the sidecar built by tools/videophy2_length_index.py.

    Entries whose length is ``null`` (unknown video spec) are dropped, so a sample
    the index cannot classify makes its whole block fall back to the original
    order rather than being sorted against a guessed length.
    """
    with open(path) as f:
        payload = json.load(f)
    raw = payload.get("lengths", payload)
    return {k: int(v) for k, v in raw.items() if v is not None}


def balance_indices(
    indices: Sequence[int],
    key_of: Callable[[int], Any],
    lengths: dict[str, int],
    world_size: int,
    grad_accum: int,
    num_workers: int,
    block_offset: int = 0,
) -> list[int]:
    """Permute `indices` within each optimizer-step block to equalise length.

    Args:
        indices: index order for ONE epoch (the unbalanced order).
        key_of: index -> manifest key, used to look the length up.
        lengths: key -> token length, from :func:`load_length_index`.
        world_size: DP world size (ranks).
        grad_accum: microsteps per optimizer step.
        num_workers: dataloader workers per rank.
        block_offset: where this epoch's element 0 sits relative to the GLOBAL
            step grid. Must be ``(-sum_of_previous_epoch_lengths) % block``;
            see :func:`epoch_block_offset`. Zero is only correct for the first
            epoch, or when the epoch length is a multiple of ``block``.

    Returns:
        A list that is a permutation of `indices`. Only blocks that align with
        the global step grid are permuted; the leading partial block, the
        trailing partial block, and any block containing an unknown-length
        sample are left in stock order.
    """
    block = world_size * grad_accum
    out = list(indices)
    start0 = block_offset % block
    for start in range(start0, len(out) - block + 1, block):
        chunk = out[start : start + block]
        try:
            keyed = [(lengths[str(key_of(i))], i) for i in chunk]
        except KeyError:
            continue  # unknown length in this block -> keep the original order
        # Sort by (length, index): the index tie-break keeps the permutation
        # deterministic and identical on every rank, which is what lets each rank
        # compute the same placement with no communication.
        keyed.sort()
        ordered = [i for _, i in keyed]
        placed = [None] * block
        for k in range(grad_accum):
            group = ordered[k * world_size : (k + 1) * world_size]
            for r in range(world_size):
                placed[slot_offset(r, k, world_size, num_workers)] = group[r]
        assert None not in placed, "slot_offset did not cover the block exactly"
        out[start : start + block] = placed
    return out


def epoch_block_offset(epoch: int, epoch_len: int, world_size: int, grad_accum: int) -> int:
    """Where epoch `epoch`'s element 0 sits relative to the global step grid.

    Every rank consumes its lane as one continuous stream across epochs, so the
    global position of this epoch's element 0 is ``epoch_len * epoch``. Blocks
    line up with optimizer steps only where that global position is a multiple
    of ``world_size * grad_accum``.

    This matters whenever ``epoch_len`` is not a multiple of the block. For
    VideoPhy-2, ``3343 % 64 == 15``: ignoring the offset made the block grid
    drift 15 positions per epoch, which broke the set-invariance property at 448
    of 500 steps and eroded the throughput gain from 22.13 s to 26.56 s per
    step. See output/videophy2_sft_nano_p800/simulate_epoch_boundary.py.
    """
    block = world_size * grad_accum
    return (-epoch_len * epoch) % block



def enabled() -> bool:
    """Whether length-balanced placement is requested.

    The module default stays OFF so other recipes are untouched; the P800
    VideoPhy-2 launcher (``examples/launch_sft_videophy2_nano_p800.sh``) sets
    ``VP2_LENGTH_BALANCE=1`` by default after the formal 500-step validation,
    mirroring how ``FSDP_DELAY_RS`` is defaulted at the launcher rather than in
    generic framework code.
    """
    return os.environ.get("VP2_LENGTH_BALANCE", "0").strip().lower() not in {"0", "false", "no", ""}


def describe(
    indices: Sequence[int],
    key_of,
    lengths,
    world_size,
    grad_accum,
    num_workers,
    block_offset: int = 0,
) -> str:
    """One-line summary of per-microstep length spread, for startup logging.

    Reports the first block that is actually aligned with the global step grid,
    which is the first one `balance_indices` touches.
    """
    block = world_size * grad_accum
    start = block_offset % block
    if len(indices) < start + block:
        return "length-balance: epoch shorter than one aligned block"
    chunk = indices[start : start + block]
    try:
        per_slot = {i: lengths[str(key_of(i))] for i in chunk}
    except KeyError:
        return "length-balance: first aligned block has unknown lengths (fell back)"
    spreads, homo = [], 0
    for k in range(grad_accum):
        vals = [per_slot[chunk[slot_offset(r, k, world_size, num_workers)]] for r in range(world_size)]
        spreads.append(max(vals) - min(vals))
        homo += int(max(vals) == min(vals))
    return (
        f"length-balance first aligned block (offset={start}): per-microstep spread "
        f"sum={sum(spreads)} homogeneous={homo}/{grad_accum} spreads={spreads}"
    )
