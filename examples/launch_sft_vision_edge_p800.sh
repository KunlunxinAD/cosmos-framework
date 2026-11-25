#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# P800 launcher for Cosmos3-Edge vision SFT. Mirrors
# launch_sft_vision_nano_p800.sh while selecting the Edge recipe and local
# Cosmos3-Edge DCP/processor snapshot.

TOML_FILE="examples/toml/sft_config/vision_sft_edge.toml"
: "${DATASET_PATH:=/workspace/dataset/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=/workspace/dataset/Cosmos3-Edge-DCP}"
: "${WAN_VAE_PATH:=/workspace/dataset/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${EDGE_PROCESSOR_PATH:=/workspace/dataset/Cosmos3-Edge}"
: "${OUTPUT_ROOT:=outputs/vision_sft_edge/vision_sft_edge_$(date +%Y%m%d_%H%M%S)}"

export COSMOS_HF_LOCAL__nvidia__Cosmos3_Edge="$EDGE_PROCESSOR_PATH"
export COSMOS_HF_LOCAL__Wan_AI__Wan2_2_TI2V_5B="$(dirname "$WAN_VAE_PATH")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_TRAINING=1
export PYTHONHASHSEED=42
export COSMOS_ATTN_BACKENDS=flash2

export XDNN_USE_FAST_SWISH=1

export COSMOS_FUSED_RMSNORM=1
export COSMOS_FUSED_ROPE=1
export COSMOS_FUSED_CAUSAL_CONV_PAD=1
export COSMOS_FUSED_VAE_NORM=1

export COSMOS_SKIP_FINAL_CHECKPOINT=1
export COSMOS_VAE_ENCODE_BATCH_SIZE=8
export COSMOS_VAE_ENCODER_NCDHW=1
export COSMOS_RELEASE_RAW_VISION_AFTER_TOKENIZATION=1
export COSMOS_INPLACE_FP32_VISION_NORMALIZATION=1


TAIL_OVERRIDES=(
    "model.config.tokenizer.dtype=float16"
    "model.config.compile.enabled=false"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
    "dataloader_train.enable_async_batch_building=true"
    "dataloader_train.dataloader.prefetch_factor=32"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
