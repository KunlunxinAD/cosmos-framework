#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for DROID action-policy SFT on Cosmos3-Nano (8B MoT).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_droid_nano.toml (selects the
# registered `action_policy_droid_nano` experiment; res480, joint_pos 8D +
# use_state, trains the generation + action heads). See
# docs/action_policy_droid_posttrain.md.
#
# Env vars (override for your filesystem):
#   DATASET_PATH          Cosmos3-DROID download dir; must be named droid_plus_lerobot_640x360_20260412
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   WANDB_API_KEY         for online logging (TOML wandb_mode="online")
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides (e.g. the keep-ranges filter)
#
# Single-node smoke (config/data sanity, a few iters):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 \
#                                dataloader_train.max_samples_per_batch=32"
#   bash examples/launch_sft_action_policy_droid_nano.sh
#
# Multi-node: launch on every worker; the trainer reads torchrun's
# --nnodes/--node_rank. For HSDP set
# model.parallelism.data_parallel_replicate_degree = <num_nodes> (shard stays 8).
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_droid_nano.toml"
: "${DATASET_PATH:=examples/data/Cosmos3-DROID/droid_plus_lerobot_640x360_20260412}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=examples/Qwen3-VL-8B-Instruct}"
: "${OUTPUT_ROOT:=outputs/action_policy_droid_nano_$(date +%Y%m%d_%H%M%S)}"
: "${NPROC_PER_NODE:=8}"
: "${ACTIVATION_CHECKPOINTING_MODE:=selective}"
: "${ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX:=["fmha", "flash_attn", "flash_attention"]}"

export DROID_ROOT="${DROID_ROOT:-$DATASET_PATH}"

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
    "trainer.callbacks.compile_tokenizer.enabled=false"
    "model.config.parallelism.data_parallel_shard_degree=8"
    "model.config.parallelism.data_parallel_replicate_degree=1"
    "dataloader_train.dataloader.num_workers=8"
    "dataloader_train.dataloader.persistent_workers=true"
    "dataloader_train.dataloader.prefetch_factor=2"
    "dataloader_train.dataloader.batch_size=16"
    "dataloader_train.max_samples_per_batch=16"
    "trainer.grad_accum_iter=1"
    "dataloader_train.dataloader.datasets.droid.dataset.video_backend=pyav"
    "model.config.activation_checkpointing.mode=$ACTIVATION_CHECKPOINTING_MODE"
    "model.config.activation_checkpointing.save_ops_regex=$ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX"
    "model.config.tokenizer.dtype=float16"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
