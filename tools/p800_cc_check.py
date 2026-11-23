# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""P800 collective-health gate: measure ring bandwidth, not just correctness.

A degraded XLink/C2C link still answers a 1 KB ``all_reduce`` correctly -- it just
falls back to PCIe (measured 18.7 GB/s vs 48.1 GB/s healthy), and a training run
started on such a link dies within tens of steps with ``KL_XID_C2C_EVENT`` retries
and a 600 s `_ALLGATHER_BASE` timeout. So the gate is bandwidth + dmesg, mirroring
``/home/qwen/preflight_cc.sh`` from the qwen3.8-27b P800 runbook.

Run under torchrun with one process per XPU::

    torchrun --nproc_per_node=8 tools/p800_cc_check.py
    # ALLREDUCE 48.1 GB/s ...
    # ALLGATHER 62.8 GB/s ...

``torch.cuda.set_device(LOCAL_RANK)`` is mandatory: without it every rank lands on
its local device 0, the collective waits on itself, and the probe reports a bogus
"link failure" (this exact false alarm happened once).
"""

import os
import time

import torch
import torch.distributed as dist


def _bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    dist.barrier()
    start = time.time()
    for _ in range(iters):
        fn()
    dist.barrier()
    return (time.time() - start) / iters


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    world = dist.get_world_size()
    n = 128 * 1024 * 1024  # 128M bf16 = 256 MB
    x = torch.ones(n, device="cuda", dtype=torch.bfloat16)

    dt = _bench(lambda: dist.all_reduce(x))
    bw = n * x.element_size() * 2 * (world - 1) / world / dt / 1e9
    if rank == 0:
        print(f"ALLREDUCE {bw:.1f} GB/s  {dt:.4f} s/iter", flush=True)

    y = torch.ones(n // world, device="cuda", dtype=torch.bfloat16)
    dt = _bench(lambda: dist.all_gather_into_tensor(x, y))
    bw = n * x.element_size() / dt / 1e9
    if rank == 0:
        print(f"ALLGATHER {bw:.1f} GB/s  {dt:.4f} s/iter", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
