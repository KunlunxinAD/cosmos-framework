# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import gc
import os
from functools import wraps

import torch_xmlir._XMLIRC as XMLIR_C
from loguru import logger as logging


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
        cpu_count = os.cpu_count() or 1
        device_count = XMLIR_C.xpumlDeviceGetCount()
        if device_count <= 0 or cpu_count <= device_count:
            return list(range(cpu_count))

        cores_per_device = max(1, cpu_count // device_count)
        start_core = self.device_idx * cores_per_device
        if self.device_idx == device_count - 1:
            end_core = cpu_count
        else:
            end_core = min(start_core + cores_per_device, cpu_count)
        return list(range(start_core, end_core))


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
