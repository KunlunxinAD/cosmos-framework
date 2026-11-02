#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# P800 launcher for Cosmos3-Super vision SFT. Uses the Super LoRA recipe with
# the local Cosmos3-Super DCP checkpoint and processor snapshot.

TOML_FILE="examples/toml/sft_config/vision_sft_super.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Super-DCP}"
: "${WAN_VAE_PATH:=examples/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${SUPER_PROCESSOR_PATH:=examples/checkpoints/Cosmos3-Super}"
: "${OUTPUT_ROOT:=outputs/vision_sft_super/vision_sft_super_$(date +%Y%m%d_%H%M%S)}"
: "${ACTIVATION_CHECKPOINTING_MODE:=selective}"
: "${ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX:=["fmha"]}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_TRAINING=1
export I4_ATTN_BACKENDS=flash2
export PYTHONHASHSEED=42

export XDNN_USE_FAST_SWISH=1
export COSMOS_FUSED_RMSNORM=1
export COSMOS_FUSED_ROPE=1
export COSMOS_FUSED_CAUSAL_CONV_PAD=1
export COSMOS_FUSED_VAE_NORM=1

TAIL_OVERRIDES=(
    "model.config.vlm_config.tokenizer.pretrained_model_name=$SUPER_PROCESSOR_PATH"
    "model.config.tokenizer.dtype=float16"
    "model.config.compile.enabled=false"
    "model.config.activation_checkpointing.mode=$ACTIVATION_CHECKPOINTING_MODE"
    "model.config.activation_checkpointing.save_ops_regex=$ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
    "model.config.parallelism.data_parallel_shard_degree=4"
    "model.config.parallelism.data_parallel_replicate_degree=2"
    +"model.config.parallelism.fsdp_frozen_param_dtype=bfloat16"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
