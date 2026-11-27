#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for action_fd_droid_posttrain.
#
# This trains forward dynamics on the Cosmos3-DROID success + failure splits. See
# docs/action_fd_droid_posttrain.md.
#
# Env vars (override for your filesystem):
#   DATASET_PATH                 Cosmos3-DROID parent dir (success/ + failure/)
#   BASE_CHECKPOINT_PATH         Base DCP checkpoint
#   WAN_VAE_PATH                 Wan2.2 VAE .pth
#   WANDB_API_KEY                for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE               torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES         space-separated Hydra overrides
#
# Single-node smoke:
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10"
#   bash examples/launch_sft_action_fd_droid_posttrain.sh
#
# Multi-node: launch on every worker. For HSDP set
# model.parallelism.data_parallel_replicate_degree = <num_nodes> (shard stays 8).
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_fd_droid_posttrain.toml"
: "${DATASET_PATH:=/workspace/dataset/droid_plus_lerobot_640x360_20260412}"
: "${BASE_CHECKPOINT_PATH:=/workspace/dataset/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/workspace/dataset/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=/workspace/dataset/Qwen3-VL-8B-Instruct}"
: "${OUTPUT_ROOT:=outputs/action_fd_droid_posttrain/action_fd_droid_posttrain_$(date +%Y%m%d_%H%M%S)}"

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
    "trainer.straggler_detection.enabled=false"
    "trainer.callbacks.compile_tokenizer.enabled=false"
    "model.config.tokenizer.dtype=float16"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
    "dataloader_train.enable_async_batch_building=true"
    "dataloader_train.dataloader.prefetch_factor=32"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
