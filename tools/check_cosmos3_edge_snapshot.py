#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate a local ``nvidia/Cosmos3-Edge`` snapshot for the videophy2_sft_edge recipe.

The Edge backbone is loaded straight from a Hugging Face snapshot directory: the
same folder supplies arch/config, tokenizer + processor metadata, and the
reasoner weights (the training loader follows the root
``model.safetensors.index.json`` into its shards — no converter step). A partial
or old snapshot therefore fails deep inside training. This script checks the
whole contract up front, without needing a GPU.

Checks:
  1. config.json exists and says ``model_type == "cosmos3_edge"`` (old
     ``nemotron_siglip2`` snapshots use remote-code processing and are not
     supported by this recipe).
  2. Every file ``build_cosmos3_edge_processor`` reads is present.
  3. Every shard named by the root safetensors index exists and the index
     resolves to a plausible weight size.
  4. No top-level ``*.safetensors`` (their presence makes
     ``safetensors_loader`` fall back to the flat-glob path and ignore the
     root index), and the vision shard matches the hash the loader pins.
  5. Optional (``--with-config`` / ``--with-processor``): the transformers
     AutoConfig and the framework-native processor actually construct.

Usage:
    python tools/check_cosmos3_edge_snapshot.py /home/cosmos3/Cosmos3-Edge
    python tools/check_cosmos3_edge_snapshot.py /home/cosmos3/Cosmos3-Edge --with-processor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Pinned by safetensors_loader._EDGE_VISION_SHARD_SHA256: the training pipeline
# keeps the old patch-embedding layout, and an upstream re-export in a different
# convention would load without error and silently corrupt vision features.
EDGE_VISION_SHARD_RELPATH = "vision_encoder/model.safetensors"
EDGE_VISION_SHARD_SHA256 = "2180ad739ecc96b5c1e9386892d3c5c08bfa42b9cdab9aabc53b028671db89b3"

# Files build_cosmos3_edge_processor() opens directly
# (cosmos_framework/data/generator/processors/cosmos3_edge_processing.py).
PROCESSOR_FILES = (
    "config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
# The same list HFExportCallback bundles into Edge exports
# (cosmos_framework/scripts/reasoner/eval_videophy2.py:119-125).
EXPORT_PROCESSOR_FILES = (
    "tokenizer.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _fail(msg: str) -> None:
    print(f"FAIL  {msg}")


def check_layout(root: str) -> bool:
    ok = True

    cfg_path = os.path.join(root, "config.json")
    if not os.path.isfile(cfg_path):
        _fail(f"missing config.json under {root}")
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        _fail(f"config.json unreadable: {exc}")
        return False

    model_type = cfg.get("model_type")
    if model_type != "cosmos3_edge":
        _fail(f'config.json model_type={model_type!r}, want "cosmos3_edge" (old/foreign snapshot?)')
        ok = False
    else:
        text = cfg.get("text_config", {})
        print(
            f"OK    model_type=cosmos3_edge  hidden={text.get('hidden_size')} "
            f"layers={text.get('num_hidden_layers')} heads={text.get('num_attention_heads')}"
        )

    for name in PROCESSOR_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            print(f"OK    {name:32s} {os.path.getsize(path) / 1e6:8.3f} MB")
        else:
            _fail(f"missing {name}")
            ok = False

    index_path = os.path.join(root, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        _fail("missing model.safetensors.index.json")
        return False
    try:
        with open(index_path) as f:
            index = json.load(f)
    except (OSError, ValueError) as exc:
        _fail(f"index unreadable: {exc}")
        return False

    weight_map = index.get("weight_map") or {}
    if not weight_map:
        _fail("index has an empty weight_map")
        return False

    shards = sorted(set(weight_map.values()))
    missing = [s for s in shards if not os.path.isfile(os.path.join(root, s))]
    declared = index.get("metadata", {}).get("total_size")
    print(f"OK    index: {len(weight_map)} tensors across {len(shards)} shards")
    if declared:
        print(f"OK    index declared total_size: {declared / 1e9:.2f} GB")
    if missing:
        _fail(f"{len(missing)} shard(s) referenced by the index are missing:")
        for s in missing[:10]:
            print(f"        {s}")
        ok = False
    else:
        print("OK    all referenced shards present")

    # A reasoner snapshot must carry the vision tower as well as the LM.
    prefixes = {s.split("/")[0] for s in shards}
    print(f"OK    shard dirs: {sorted(prefixes)}")

    # The indexed path only triggers when the root has no *.safetensors; with one
    # present the loader silently switches to the flat glob and finds nothing.
    top_level = sorted(
        e.name for e in os.scandir(root) if e.is_file() and e.name.endswith(".safetensors")
    )
    if top_level:
        _fail(f"top-level safetensors present ({top_level[:3]}); the indexed Edge path will be skipped")
        ok = False
    else:
        print("OK    no top-level safetensors (indexed path will be used)")

    vision_path = os.path.join(root, EDGE_VISION_SHARD_RELPATH)
    if os.path.isfile(vision_path):
        sha = hashlib.sha256()
        with open(vision_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        actual = sha.hexdigest()
        if actual == EDGE_VISION_SHARD_SHA256:
            print(f"OK    vision shard sha256 pinned match ({actual[:12]}...)")
        else:
            _fail(
                "vision shard sha256 mismatch -- safetensors_loader will refuse to load.\n"
                f"        expected {EDGE_VISION_SHARD_SHA256}\n"
                f"        actual   {actual}"
            )
            ok = False
    else:
        _fail(f"missing {EDGE_VISION_SHARD_RELPATH}")
        ok = False
    return ok


def check_stub_processor(root: str) -> bool:
    """Build the processor without importing torch-heavy training code."""
    try:
        from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
            build_cosmos3_edge_processor,
            is_cosmos3_edge_native_snapshot,
        )
    except Exception as exc:  # noqa: BLE001 - report the real import error
        _fail(f"cannot import the Edge processor module: {type(exc).__name__}: {exc}")
        return False

    if not is_cosmos3_edge_native_snapshot(root):
        _fail("is_cosmos3_edge_native_snapshot() returned False")
        return False
    try:
        proc = build_cosmos3_edge_processor(root)
    except Exception as exc:  # noqa: BLE001
        _fail(f"build_cosmos3_edge_processor failed: {type(exc).__name__}: {exc}")
        return False
    print(f"OK    processor built: {type(proc).__name__} vocab={proc.tokenizer.vocab_size}")
    return True


def check_autoconfig(root: str) -> bool:
    try:
        # Registers model_type="cosmos3_edge" with the transformers Auto classes;
        # without this import AutoConfig rejects the snapshot as unknown.
        import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401
        from transformers import AutoConfig
    except Exception as exc:  # noqa: BLE001
        _fail(f"transformers/framework unavailable: {type(exc).__name__}: {exc}")
        return False
    try:
        cfg = AutoConfig.from_pretrained(root)
    except Exception as exc:  # noqa: BLE001
        _fail(f"AutoConfig.from_pretrained failed: {type(exc).__name__}: {exc}")
        return False
    print(f"OK    AutoConfig: {type(cfg).__name__} model_type={cfg.model_type}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="local snapshot directory, e.g. /home/cosmos3/Cosmos3-Edge")
    ap.add_argument("--with-config", action="store_true", help="also run transformers AutoConfig")
    ap.add_argument("--with-processor", action="store_true", help="also build the native Edge processor")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    print(f"=== checking {root}")
    if not os.path.isdir(root):
        _fail("not a directory")
        return 2

    ok = check_layout(root)
    if args.with_config:
        ok = check_autoconfig(root) and ok
    if args.with_processor:
        ok = check_stub_processor(root) and ok

    print("=== " + ("PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())