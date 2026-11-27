#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Usage: bash examples/launch_sft_sim_bimanual_p800.sh {0|1|2|3} [key=value ...]
set -uo pipefail

stage=$1
shift
stage_names=(stage0_bidir stage1_tf stage2_tfdcm stage3_sfdmd)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.." || exit 1
TOML_FILE="examples/toml/sft_config/sim_bimanual_${stage_names[$stage]}.toml"
: "${DATASET_PATH:=/workspace/dataset/sim_bimanual_robot_example_v1}"
: "${BASE_CHECKPOINT_PATH:=/workspace/dataset/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/workspace/dataset/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
: "${OUTPUT_ROOT:=outputs/sim_bimanual/${stage_names[$stage]}_$(date +%Y%m%d_%H%M%S)}"

export COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE="${DATASET_PATH}"
export COSMOS_SIM_BIMANUAL_GA_CHECKPOINT="${BASE_CHECKPOINT_PATH}"

export COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT="${COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT:-/workspace/dataset/sim_bimanual_a800_stage0}"
export COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT="${COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT:-/workspace/dataset/sim_bimanual_a800_stage1}"
export COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT="${COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT:-/workspace/dataset/sim_bimanual_a800_stage2}"

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

export COSMOS_DETERMINISTIC=0
export COSMOS_NATTEN_DENSE_FALLBACK=1
export COSMOS_FLASH2_SAC_CUSTOM_OP=1
# Container torch_xmlir/libxpu_flash_attention.so produces standard LSE layouts (keep 0).
# Only set to 1 if an external/legacy flash_attn wheel providing libxpu_flash_attention_train.so is active.
export COSMOS_FLASH2_LSE_BHFIX=0
export COSMOS_FLASH2_VARLEN_PACKED_LSE=0

TAIL_OVERRIDES=(
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH"
    "model.config.tokenizer.dtype=float16"
    "trainer.seed=42"
    "model.config.activation_checkpointing.mode=selective"
    "dataloader_train.max_samples_per_batch=1"
    "trainer.logging_iter=1"
    "trainer.grad_accum_iter=1"
    "trainer.max_iter=12"
    "model.config.ema.enabled=false"
)
if [[ "$stage" != "3" ]]; then
    TAIL_OVERRIDES+=("trainer.callbacks.compile_tokenizer.enabled=false")
fi
TAIL_OVERRIDES+=("$@")

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
