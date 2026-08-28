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
: "${DATASET_PATH:=examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=examples/Qwen3-VL-8B-Instruct}"
: "${OUTPUT_ROOT:=outputs/vision_sft_nano_$(date +%Y%m%d_%H%M%S)}"
: "${NPROC_PER_NODE:=8}"
: "${MAX_ITER:=500}"
: "${TRAIN_PRECISION:=bfloat16}"
: "${VAE_DTYPE:=float16}"
: "${FSDP_MASTER_DTYPE:=bfloat16}"
#Packed QKV And Gate_Up
: "${PACKED_QKV:=false}"
: "${PACKED_GATE_UP:=false}"
#Partial Activation Recomputation
: "${ACTIVATION_CHECKPOINTING_MODE:=selective}"
: "${ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX:=["fmha", "flash_attn", "flash_attention"]}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_TRAINING=1
export I4_ATTN_BACKENDS=flash2
export PYTHONHASHSEED=42

export XDNN_USE_FAST_SWISH=1
export XDNN_USE_FAST_GELU=1

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDART_DUMMY_REGISTER=1
unset CUDA_LAUNCH_BLOCKING
export XMLIR_ENABLE_LINEAR_FC_FUSION=1
export XMLIR_ENABLE_FAST_FC=1
export XPYTORCH_RUN_ENHANCE=1
export XDNN_FAST_DIV_SCALAR=true
export XPUAPI_SDNN_BF16_ROUND_MODE=3

TAIL_OVERRIDES=(
    "model.config.vlm_config.tokenizer.pretrained_model_name=$TOKENIZER_PATH"
    "model.config.precision=$TRAIN_PRECISION"
    "model.config.tokenizer.dtype=$VAE_DTYPE"
    "model.config.parallelism.fsdp_master_dtype=$FSDP_MASTER_DTYPE"
    "model.config.compile.enabled=false"
    # "+model.config.vlm_config.model_instance.config.packed_qkv=$PACKED_QKV"
    # "+model.config.vlm_config.model_instance.config.packed_gate_up=$PACKED_GATE_UP"
    # "model.config.activation_checkpointing.mode=$ACTIVATION_CHECKPOINTING_MODE"
    # "model.config.activation_checkpointing.save_ops_regex=$ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
    "trainer.max_iter=$MAX_ITER"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"