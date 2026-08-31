# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert an unpacked, model-only Cosmos3 DCP to packed QKV and Gate/Up."""

from cosmos_framework.inference.common.init import init_script

init_script(env={"COSMOS_DEVICE": "cpu", "COSMOS_TRAINING": "1"})

from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Annotated
import uuid

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
import tyro

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.checkpoint.packed_qkv import (
    allocate_state_dict_from_dcp_metadata,
    pack_gate_up_state_dict,
    pack_qkv_state_dict,
    state_dict_nbytes,
)


@dataclass
class Args:
    source_path: Annotated[Path, tyro.conf.arg(aliases=("-i",))]
    """Read-only source DCP root, or its model/ directory."""
    output_path: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """New DCP root. It must not already exist."""
    max_shard_size_gib: float = 5.0
    """Approximate maximum output distcp shard size."""


def _resolve_source(source_path: Path) -> tuple[Path, Path]:
    source_path = source_path.resolve()
    if (source_path / "model" / ".metadata").is_file():
        return source_path, source_path / "model"
    if source_path.name == "model" and (source_path / ".metadata").is_file():
        return source_path.parent, source_path
    raise FileNotFoundError(f"No DCP model/.metadata found under {source_path}")


def _validate_paths(source_root: Path, output_root: Path) -> None:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output path: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError(f"Output path must be outside the source DCP tree: {output_root}")
    if output_root in source_root.parents:
        raise ValueError(f"Output path may not contain the source DCP tree: {output_root}")


def _snapshot_source(source_root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(source_root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in source_root.rglob("*")
        if path.is_file()
    }


def _write_packed_config(source_model_dir: Path, output_model_dir: Path) -> None:
    source_config = source_model_dir / "config.json"
    if not source_config.is_file():
        return
    config = json.loads(source_config.read_text())
    try:
        model_config = config["model"]["config"]["vlm_config"]["model_instance"]["config"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unexpected Cosmos3 config layout in {source_config}") from error
    model_config["packed_qkv"] = True
    model_config["packed_gate_up"] = True
    (output_model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _validate_output(
    output_model_dir: Path,
    packed_state: dict[str, torch.Tensor],
    packed_keys: tuple[str, ...],
) -> None:
    reader = FileSystemReader(output_model_dir)
    metadata = reader.read_metadata()
    if set(metadata.state_dict_metadata) != set(packed_state):
        missing = sorted(set(packed_state) - set(metadata.state_dict_metadata))
        unexpected = sorted(set(metadata.state_dict_metadata) - set(packed_state))
        raise ValueError(f"Output DCP metadata mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")

    for key, tensor in packed_state.items():
        tensor_metadata = metadata.state_dict_metadata[key]
        if tuple(tensor_metadata.size) != tuple(tensor.shape):
            raise ValueError(f"Output shape mismatch for {key}: {tensor_metadata.size} != {tensor.shape}")
        if tensor_metadata.properties.dtype != tensor.dtype:
            raise ValueError(f"Output dtype mismatch for {key}: {tensor_metadata.properties.dtype} != {tensor.dtype}")

    sample_keys = tuple(dict.fromkeys((packed_keys[0], packed_keys[len(packed_keys) // 2], packed_keys[-1])))
    sample_state = OrderedDict((key, torch.empty_like(packed_state[key])) for key in sample_keys)
    dcp.load(
        state_dict=sample_state,
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
        no_dist=True,
    )
    for key in sample_keys:
        torch.testing.assert_close(sample_state[key], packed_state[key], rtol=0, atol=0)


def convert_dcp_to_packed_qkv(args: Args) -> None:
    source_root, source_model_dir = _resolve_source(args.source_path)
    output_root = args.output_path.resolve()
    _validate_paths(source_root, output_root)
    if args.max_shard_size_gib <= 0:
        raise ValueError("max_shard_size_gib must be positive")

    source_snapshot = _snapshot_source(source_root)
    temporary_root = output_root.with_name(f".{output_root.name}.tmp-{uuid.uuid4().hex}")
    temporary_model_dir = temporary_root / "model"
    temporary_root.mkdir(parents=True)

    print(f"Reading source DCP without modifying it: {source_root}")
    source_reader = FileSystemReader(source_model_dir)
    source_metadata = source_reader.read_metadata()
    source_state = allocate_state_dict_from_dcp_metadata(source_metadata)
    dcp.load(state_dict=source_state, storage_reader=source_reader, no_dist=True)

    print("Packing Q/K/V and Gate/Up projection tensors...")
    qkv_packed_state, qkv_report = pack_qkv_state_dict(source_state)
    packed_state, gate_up_report = pack_gate_up_state_dict(qkv_packed_state)
    source_bytes = state_dict_nbytes(source_state)
    output_bytes = state_dict_nbytes(packed_state)
    if source_bytes != output_bytes:
        raise ValueError(f"Tensor byte count changed during conversion: {source_bytes} != {output_bytes}")

    max_shard_size = int(args.max_shard_size_gib * 1024**3)
    thread_count = max(1, math.ceil(output_bytes / max_shard_size))
    print(f"Saving {output_bytes / 1024**3:.2f} GiB packed DCP with {thread_count} writer threads...")
    dcp.save(
        state_dict=packed_state,
        storage_writer=FileSystemWriter(temporary_model_dir, thread_count=thread_count),
        planner=CustomSavePlanner(),
    )

    checkpoint_json = source_root / "checkpoint.json"
    if checkpoint_json.is_file():
        shutil.copy2(checkpoint_json, temporary_root / "checkpoint.json")
    _write_packed_config(source_model_dir, temporary_model_dir)

    conversion_record = {
        "format": "cosmos3-packed-qkv-gate-up-dcp-v1",
        "source_path": str(source_root),
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "source_nbytes": source_bytes,
        "output_nbytes": output_bytes,
        "qkv": asdict(qkv_report),
        "gate_up": asdict(gate_up_report),
    }
    (temporary_root / "packed_projection_conversion.json").write_text(json.dumps(conversion_record, indent=2) + "\n")

    print("Validating output metadata and sampled packed tensors...")
    _validate_output(temporary_model_dir, packed_state, qkv_report.packed_keys + gate_up_report.packed_keys)
    if _snapshot_source(source_root) != source_snapshot:
        raise RuntimeError(f"Source DCP changed during conversion: {source_root}")

    os.replace(temporary_root, output_root)
    print(
        f"Converted {qkv_report.source_qkv_tensor_count} Q/K/V tensors into "
        f"{qkv_report.packed_tensor_count} packed tensors and "
        f"{gate_up_report.source_gate_up_tensor_count} Gate/Up tensors into "
        f"{gate_up_report.packed_tensor_count} packed tensors."
    )
    print(f"Saved packed DCP to {output_root}")


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    convert_dcp_to_packed_qkv(args)


if __name__ == "__main__":
    main()
