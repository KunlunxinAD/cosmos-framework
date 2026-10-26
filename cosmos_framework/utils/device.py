# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import gc
import os
from functools import wraps

import torch_xmlir._XMLIRC as XMLIR_C
from loguru import logger as logging

from cosmos_framework.utils.cpu_affinity import initial_cpu_affinity, numa_cpu_affinity


def get_gpu_architecture() -> str:
    """
    Retrieves the GPU architecture of the available GPUs.

    Returns:
        str: The GPU architecture, which can be "H100", "A100", or "Other".
    """
    xpuml_initialized = False
    try:
        XMLIR_C.xpumlInit()
        xpuml_initialized = True
        device_count = XMLIR_C.xpumlDeviceGetCount()
        for i in range(device_count):
            handle = XMLIR_C.xpumlDeviceGetHandleByIndex(i)
            model_name = XMLIR_C.xpumlDeviceGetName(handle)
            if isinstance(model_name, bytes):
                model_name = model_name.decode("utf-8")
            if "H100" in model_name or "H200" in model_name:
                return "H100"
            elif "A100" in model_name:
                return "A100"
            elif "L40S" in model_name:
                return "L40S"
            elif "B200" in model_name:
                return "B200"
    except Exception as error:
        logging.error(f"Error retrieving device information: {error}")
        return "Other"
    finally:
        if xpuml_initialized:
            XMLIR_C.xpumlShutdown()

    # return "Other" incase of non hopper/ampere or error
    return "Other"


class GPUArchitectureNotSupported(Exception):
    """
    Custom exception raised when the expected GPU architecture is not supported.
    """

    pass


def print_gpu_mem(str=None):
    try:
        XMLIR_C.xpumlInit()
        meminfo = XMLIR_C.xpumlDeviceGetMemoryInfo(XMLIR_C.xpumlDeviceGetHandleByIndex(0))
        logging.info(
            f"{str}: {meminfo.used / 1024 / 1024}/{meminfo.total / 1024 / 1024}MiB used ({meminfo.free / 1024 / 1024}MiB free)"
        )
    except Exception as error:
        print(f"Failed to get device memory info: {error}")


def force_gc():
    print_gpu_mem()
    print("gc()")
    gc.collect()
    print_gpu_mem()
    print("empty cuda cache")
    # print(torch.cuda.memory_summary())
    print_gpu_mem()


def gpu0_has_80gb_or_less():
    try:
        XMLIR_C.xpumlInit()
        meminfo = XMLIR_C.xpumlDeviceGetMemoryInfo(XMLIR_C.xpumlDeviceGetHandleByIndex(0))
        return meminfo.total / 1024 / 1024 / 1024 <= 80
    except Exception as error:
        print(f"Failed to get device memory info: {error}")


class Device:
    def __init__(self, device_idx: int):
        super().__init__()
        self.device_idx = device_idx
        XMLIR_C.xpumlInit()
        self.handle = XMLIR_C.xpumlDeviceGetHandleByIndex(device_idx)

    def get_name(self) -> str:
        return XMLIR_C.xpumlDeviceGetName(self.handle)

    def get_cpu_affinity(self) -> list[int]:
        """Use physical PCI/NUMA topology, honoring CUDA visibility and the allocated cpuset."""
        import torch

        allocated_cpus = initial_cpu_affinity(os.getpid())
        try:
            visible_count = torch.cuda.device_count()
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(visible_count)))
            if not 0 < local_world_size <= visible_count:
                raise ValueError(f"LOCAL_WORLD_SIZE={local_world_size}, visible devices={visible_count}")
            pci_bus_ids = []
            for index in range(local_world_size):
                props = torch.cuda.get_device_properties(index)
                # The runtime resolves CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER.
                # Management-library indices are machine-wide and must not use local_rank.
                pci_bus_ids.append(f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0")
            node, cpus = numa_cpu_affinity(self.device_idx, tuple(pci_bus_ids), allocated_cpus)
            logging.info(
                f"CPU affinity: local_rank={self.device_idx}, pci={pci_bus_ids[self.device_idx]}, "
                f"numa={node}, cpus={cpus}"
            )
            return cpus
        except (OSError, ValueError, AttributeError, RuntimeError) as error:
            logging.warning(f"Cannot determine GPU-local CPU affinity; retaining allocated CPUs: {error}")
            return list(allocated_cpus)


def with_torch_device(device):
    """
    Decorator factory that wraps a function to execute within a specific torch device context.

    This decorator ensures that all tensor allocations and operations within the decorated
    function use the specified device by default.

    Args:
        device: The torch device to use (e.g., 'cuda', 'cuda:0', 'cpu', or torch.device object).

    Returns:
        A decorator function that wraps the target function with the specified device context.

    Example:
        @with_torch_device('cuda:0')
        def create_tensors():
            x = torch.randn(10, 10)  # Will be created on cuda:0
            return x
    """
    import torch

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.device(device):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
