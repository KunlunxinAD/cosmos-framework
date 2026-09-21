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
#
# Pre-sync the 20 FPS suite once:
#   hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include 'libero_10/**' --local-dir <dir>
#   export LIBERO_ROOT=<dir>/libero_10
#
# Usage (HSDP 2x8; set NNODES/NODE_RANK/MASTER_ADDR per node):
#   LIBERO_ROOT=<dir>/libero_10 bash examples/launch_sft_action_policy_libero_10_nano.sh

TOML_FILE="examples/toml/sft_config/action_policy_libero_10_nano.toml"
: "${DATASET_PATH:=examples/data/LIBERO_LeRobot_v3/libero_10}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=examples/Qwen3-VL-8B-Instruct}"
: "${OUTPUT_ROOT:=outputs/action_policy_libero_10_nano_$(date +%Y%m%d_%H%M%S)}"
: "${ACTIVATION_CHECKPOINTING_MODE:=selective}"
: "${ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX:=["fmha", "flash_attn", "flash_attention"]}"

export LIBERO_ROOT="${LIBERO_ROOT:-$DATASET_PATH}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
    "model.config.compile.enabled=false"
    "model.config.parallelism.data_parallel_shard_degree=8"
    "model.config.parallelism.data_parallel_replicate_degree=1"
    "model.config.activation_checkpointing.mode=$ACTIVATION_CHECKPOINTING_MODE"
    "model.config.activation_checkpointing.save_ops_regex=$ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX"
    "model.config.tokenizer.dtype=float16"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
