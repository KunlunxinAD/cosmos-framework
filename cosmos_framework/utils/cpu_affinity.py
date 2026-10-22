# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Linux CPU placement for the CUDA-visible, node-local training ranks."""

import os
from functools import lru_cache
from pathlib import Path

PCI_DEVICES = Path("/sys/bus/pci/devices")
NUMA_NODES = Path("/sys/devices/system/node")
CPU_DEVICES = Path("/sys/devices/system/cpu")


def parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        bounds = part.split("-")
        first, last = int(bounds[0]), int(bounds[-1])
        cpus.update(range(first, last + 1))
    return sorted(cpus)


@lru_cache(maxsize=1)
def initial_cpu_affinity(pid: int) -> tuple[int, ...]:
    """Snapshot before our first bind; PID prevents inheriting a parent's cache after fork."""
    return tuple(sorted(os.sched_getaffinity(0)))


def device_numa_node(pci_bus_id: str) -> int:
    node = int((PCI_DEVICES / pci_bus_id / "numa_node").read_text())
    if node < 0:
        raise ValueError(f"Unknown NUMA node for PCI device {pci_bus_id}")
    return node


def _physical_cores(cpus: set[int]) -> list[list[int]]:
    """Keep available SMT siblings together, so ranks do not share a physical core."""
    cores: dict[tuple[int, ...], list[int]] = {}
    for cpu in sorted(cpus):
        try:
            siblings = tuple(parse_cpu_list((CPU_DEVICES / f"cpu{cpu}/topology/thread_siblings_list").read_text()))
        except OSError:
            siblings = (cpu,)
        cores.setdefault(siblings, []).append(cpu)
    return [cores[key] for key in sorted(cores)]


def numa_cpu_affinity(
    device_idx: int, pci_bus_ids: tuple[str, ...], allocated_cpus: tuple[int, ...]
) -> tuple[int, list[int]]:
    """Partition local physical cores among participating ranks on the same NUMA node.

    pci_bus_ids is in logical CUDA order after CUDA_VISIBLE_DEVICES remapping,
    limited to LOCAL_WORLD_SIZE, not the machine-wide management-library count.
    """
    if not 0 <= device_idx < len(pci_bus_ids):
        raise ValueError(f"Local device {device_idx} outside participating devices {pci_bus_ids}")
    nodes = [device_numa_node(bus_id) for bus_id in pci_bus_ids]
    node = nodes[device_idx]
    peers = [index for index, peer_node in enumerate(nodes) if peer_node == node]
    local_cpus = set(parse_cpu_list((NUMA_NODES / f"node{node}/cpulist").read_text()))
    cores = _physical_cores(local_cpus.intersection(allocated_cpus))
    if not cores:
        raise ValueError(f"No allocated CPU is local to PCI device {pci_bus_ids[device_idx]} (NUMA {node})")
    position = peers.index(device_idx)
    if len(cores) < len(peers):
        # An explicitly restricted cpuset can require sharing; never return an empty mask.
        selected = [cores[position % len(cores)]]
    else:
        selected = cores[position * len(cores) // len(peers) : (position + 1) * len(cores) // len(peers)]
    return node, sorted(cpu for core in selected for cpu in core)
