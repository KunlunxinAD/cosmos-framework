#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# P800 debug launcher for the CosmosDataLoader/MapDistributor variant of
# vision_sft_nano. This recipe uses a finite map-style SFT metadata adapter,
# so the dataloader emits per-worker epoch/index state for checkpoint resume.

TOML_FILE="examples/toml/sft_config/vision_sft_nano_mapstyle_dataloader.toml"
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=examples/Qwen3-VL-8B-Instruct}"
: "${OUTPUT_ROOT:=outputs/vision_sft_nano_mapstyle_dataloader/vision_sft_nano_mapstyle_dataloader_$(date +%Y%m%d_%H%M%S)}"
: "${ACTIVATION_CHECKPOINTING_MODE:=selective}"
: "${ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX:=["fmha", "flash_attn", "flash_attention"]}"

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
    "model.config.vlm_config.tokenizer.pretrained_model_name=$TOKENIZER_PATH"
    "model.config.tokenizer.dtype=float16"
    "model.config.compile.enabled=false"
    "model.config.activation_checkpointing.mode=$ACTIVATION_CHECKPOINTING_MODE"
    "model.config.activation_checkpointing.save_ops_regex=$ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
)

LOG_FILTER_REGEX='Text ids are too long, truncating: [0-9]+ > 1024'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
