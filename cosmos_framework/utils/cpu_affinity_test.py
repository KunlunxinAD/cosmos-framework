# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.utils import cpu_affinity
from cosmos_framework.utils.device import Device


@pytest.fixture
def topology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, ...]]:
    pci = tmp_path / "pci"
    nodes = tmp_path / "nodes"
    cpus = tmp_path / "cpus"
    monkeypatch.setattr(cpu_affinity, "PCI_DEVICES", pci)
    monkeypatch.setattr(cpu_affinity, "NUMA_NODES", nodes)
    monkeypatch.setattr(cpu_affinity, "CPU_DEVICES", cpus)
    for node, value in [(0, "0-127,256-383"), (1, "128-255,384-511")]:
        p = nodes / f"node{node}"
        p.mkdir(parents=True)
        (p / "cpulist").write_text(value)
    buses = tuple(f"0000:{index + 16:02x}:00.0" for index in range(16))
    for index, bus in enumerate(buses):
        p = pci / bus
        p.mkdir(parents=True)
        (p / "numa_node").write_text(str(index // 8))
    for core in range(256):
        for cpu in [core, core + 256]:
            p = cpus / f"cpu{cpu}" / "topology"
            p.mkdir(parents=True)
            (p / "thread_siblings_list").write_text(f"{core},{core + 256}")
    cpu_affinity.initial_cpu_affinity.cache_clear()
    yield buses
    cpu_affinity.initial_cpu_affinity.cache_clear()


@pytest.mark.parametrize("offset,node", [(0, 0), (8, 1)])
def test_each_eight_card_group_gets_balanced_local_physical_cores(topology, offset, node):
    selected = topology[offset : offset + 8]
    masks = [cpu_affinity.numa_cpu_affinity(rank, selected, tuple(range(512))) for rank in range(8)]
    local = set(cpu_affinity.parse_cpu_list((cpu_affinity.NUMA_NODES / f"node{node}/cpulist").read_text()))
    assert all(numa == node and len(mask) == 32 for numa, mask in masks)
    assert set().union(*(set(mask) for _, mask in masks)) == local
    assert sum(len(mask) for _, mask in masks) == len(local)
    for _, mask in masks:
        assert len({cpu % 256 for cpu in mask}) == 16
        assert all((cpu + 256) % 512 in mask for cpu in mask)


def test_reordered_mixed_visible_devices_partition_per_numa(topology):
    selected = (topology[9], topology[1], topology[8], topology[0])
    masks = [cpu_affinity.numa_cpu_affinity(rank, selected, tuple(range(512))) for rank in range(4)]
    assert [node for node, _ in masks] == [1, 0, 1, 0]
    assert all(len(mask) == 128 for _, mask in masks)
    assert not set(masks[0][1]).intersection(masks[2][1])
    assert not set(masks[1][1]).intersection(masks[3][1])


def test_restricted_cpuset_keeps_smt_siblings_on_same_rank(topology):
    selected = topology[8:10]
    allocated = (128, 129, 130, 131, 384, 385, 386)
    masks = [cpu_affinity.numa_cpu_affinity(rank, selected, allocated)[1] for rank in range(2)]
    assert {cpu % 256 for cpu in masks[0]} == {128, 129}
    assert {cpu % 256 for cpu in masks[1]} == {130, 131}
    assert set(masks[0]) | set(masks[1]) == set(allocated)
    assert not set(masks[0]).intersection(masks[1])


def test_tiny_explicit_cpuset_shares_instead_of_returning_empty_mask(topology):
    allocated = (128, 384)
    for rank in range(8):
        assert cpu_affinity.numa_cpu_affinity(rank, topology[8:], allocated) == (1, [128, 384])


def test_no_local_allocated_cpu_does_not_guess_remote_cpu(topology):
    with pytest.raises(ValueError, match="No allocated CPU is local"):
        cpu_affinity.numa_cpu_affinity(0, topology[8:], tuple(range(128)))


def test_unknown_numa_node_does_not_guess(topology):
    (cpu_affinity.PCI_DEVICES / topology[8] / "numa_node").write_text("-1")
    with pytest.raises(ValueError, match="Unknown NUMA node"):
        cpu_affinity.numa_cpu_affinity(0, topology[8:], tuple(range(512)))


def test_original_cpuset_is_stable_after_bind_and_separate_after_fork(monkeypatch):
    cpu_affinity.initial_cpu_affinity.cache_clear()
    allowed = set(range(512))
    monkeypatch.setattr(cpu_affinity.os, "sched_getaffinity", lambda _: allowed)
    first = cpu_affinity.initial_cpu_affinity(100)
    allowed = {128, 384}
    assert cpu_affinity.initial_cpu_affinity(100) == first
    assert cpu_affinity.initial_cpu_affinity(101) == (128, 384)
    cpu_affinity.initial_cpu_affinity.cache_clear()


def test_device_uses_runtime_visible_order_and_repeated_calls_do_not_shrink(topology, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    allowed = set(range(512))
    monkeypatch.setattr(cpu_affinity.os, "sched_getaffinity", lambda _: allowed)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(pci_domain_id=0, pci_bus_id=24 + index, pci_device_id=0),
    )
    device = object.__new__(Device)
    device.device_idx = 0
    first = device.get_cpu_affinity()
    assert set(first) == set(range(128, 144)) | set(range(384, 400))
    allowed = set(first)
    assert device.get_cpu_affinity() == first
    assert cpu_affinity.initial_cpu_affinity(os.getpid()) == tuple(range(512))


def test_device_preserves_cpuset_when_pci_information_is_unavailable(topology, monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setattr(cpu_affinity.os, "sched_getaffinity", lambda _: {2, 7})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace())
    device = object.__new__(Device)
    device.device_idx = 0
    assert device.get_cpu_affinity() == [2, 7]
