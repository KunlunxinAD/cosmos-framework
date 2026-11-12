#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for llava_ov (VLM SFT on
# lmms-lab/LLaVA-OneVision-Data via CosmosDataLoader) on Kunlun P800.
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/llava_ov.toml.
#
# Usage:
#   # 8-card default launch:
#   bash examples/launch_sft_llava_ov_p800.sh

TOML_FILE="examples/toml/sft_config/llava_ov.toml"

: "${LLAVA_OV_LOCAL_DATA_DIR:=/home/projects/cosmos-framework-workspace/datasets/ai2d(gpt4v)}"
: "${MODEL_PATH:=/home/projects/cosmos-framework-workspace/models/Qwen3-VL-8B-Instruct}"
: "${VLM_SAFETENSORS_PATH:=/home/projects/cosmos-framework-workspace/models/Cosmos3-Nano-VLM}"
: "${WANDB_ENV_FILE:=/home/projects/cosmos-framework-workspace/.wandb_env}"

if [[ -f "${WANDB_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${WANDB_ENV_FILE}"
fi

export LLAVA_OV_LOCAL_DATA_DIR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

TAIL_OVERRIDES=(
    "model.config.policy.backbone.model_name=$MODEL_PATH"
    # P800 XPU runtime does not support cudaHostRegister used by DCP async pinned-memory staging
    # (fails with RuntimeError: Registering memory failed with cudaError: 2 at checkpoint save).
    "checkpoint.dcp_async_mode_enabled=false"
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    TAIL_OVERRIDES+=("job.wandb_mode=${WANDB_MODE:-online}")
fi

if [[ -n "${VLM_SAFETENSORS_PATH:-}" ]]; then
    TAIL_OVERRIDES+=("model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH")
fi

if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    set -f
    # shellcheck disable=SC2206
    TAIL_OVERRIDES+=( ${EXTRA_TAIL_OVERRIDES} )
    set +f
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
