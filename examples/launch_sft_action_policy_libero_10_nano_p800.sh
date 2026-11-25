#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_libero_nano — Cosmos3-Nano LIBERO
# action-policy SFT (HSDP, full SFT). Drives cosmos_framework.scripts.train
# against examples/toml/sft_config/action_policy_libero_10_nano.toml.
#
# Point LIBERO_ROOT at the libero_10 suite ONLY. Use the 20 FPS
# nvidia/LIBERO_LeRobot_v3. The default recipe is HSDP 2x8 (global batch 2048);
# set NNODES/NODE_RANK/MASTER_ADDR per node.
# See docs/action_policy_libero_posttrain.md.
#
# Required env vars:
#   LIBERO_ROOT           local LIBERO-10 LeRobot dataset dir, e.g. <dir>/libero_10 (no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Nano
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   HF_TOKEN              if any tokenizer download requires gated HF access
#   OUTPUT_ROOT           default: outputs/train
#   COSMOS_RELEASE_RAW_VISION_AFTER_TOKENIZATION   Set to 1 to release GPU raw vision (default: on).
#   COSMOS_INPLACE_FP32_VISION_NORMALIZATION       Set to 1 for in-place FP32 math (default: on).
#
# Pre-sync the 20 FPS suite once:
#   hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include 'libero_10/**' --local-dir <dir>
#   export LIBERO_ROOT=<dir>/libero_10
#
# Usage (HSDP 2x8; set NNODES/NODE_RANK/MASTER_ADDR per node):
#   LIBERO_ROOT=<dir>/libero_10 bash examples/launch_sft_action_policy_libero_10_nano.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_10_nano.toml"
: "${DATASET_PATH:=/workspace/dataset/LIBERO/libero_10}"
: "${BASE_CHECKPOINT_PATH:=/workspace/dataset/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/workspace/dataset/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=/workspace/dataset/Qwen3-VL-8B-Instruct}"
: "${OUTPUT_ROOT:=outputs/action_policy_libero_10/action_policy_libero_10_nano_$(date +%Y%m%d_%H%M%S)}"

export LIBERO_ROOT="${LIBERO_ROOT:-$DATASET_PATH}"

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
    "model.config.compile.enabled=false"
    "model.config.parallelism.data_parallel_shard_degree=8"
    "model.config.parallelism.data_parallel_replicate_degree=-1"
    "model.config.tokenizer.dtype=float16"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
    "dataloader_train.enable_async_batch_building=true"
    "dataloader_train.dataloader.prefetch_factor=32"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
