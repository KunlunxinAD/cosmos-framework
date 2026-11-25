#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_nano (T2V / I2V / V2V vision-only
# SFT on Qwen3-VL-8B, 8-GPU FSDP). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/vision_sft_nano.toml.
#
# Optional env vars (defaults below point under examples/; override to put
# data or checkpoints on a different filesystem):
#   DATASET_PATH          default: examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
#                         (must contain train/video_dataset_file.jsonl)
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (8-GPU allocation, inside the training container, from the repo root):
#   bash examples/launch_sft_vision_nano.sh

TOML_FILE="examples/toml/sft_config/vision_sft_nano.toml"
: "${DATASET_PATH:=/workspace/dataset/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=/workspace/dataset/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/workspace/dataset/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=/workspace/dataset/Qwen3-VL-8B-Instruct}"
: "${OUTPUT_ROOT:=outputs/vision_sft_nano/vision_sft_nano_$(date +%Y%m%d_%H%M%S)}"

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
    "model.config.vlm_config.tokenizer.pretrained_model_name=$TOKENIZER_PATH"
    "model.config.tokenizer.dtype=float16"
    "model.config.compile.enabled=false"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
    "dataloader_train.enable_async_batch_building=true"
    "dataloader_train.dataloader.prefetch_factor=32"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"