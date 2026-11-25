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

# ==================== 环境变量配置 ====================
# 本脚本默认开启已验证的 P800 优化；外部显式传入的值优先。
# Cosmos Framework 源码中的实验优化仍默认关闭，不影响其他启动入口。

# W&B 环境配置文件，用于加载认证及运行配置。
: "${WANDB_ENV_FILE:=/home/projects/cosmos-framework-workspace/.wandb_env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${WANDB_ENV_FILE}"
fi

# 本地 LLaVA-OneVision 数据集目录。
export LLAVA_OV_LOCAL_DATA_DIR="${LLAVA_OV_LOCAL_DATA_DIR:-/home/projects/cosmos-framework-workspace/datasets/ai2d(gpt4v)}"
# 本地 Qwen3-VL 模型配置与 tokenizer 路径。
: "${MODEL_PATH:=/home/projects/cosmos-framework-workspace/models/Qwen3-VL-8B-Instruct}"
# 本地 Cosmos3-Nano-VLM 权重路径。
: "${VLM_SAFETENSORS_PATH:=/home/projects/cosmos-framework-workspace/models/Cosmos3-Nano-VLM}"
# 可见的 P800 设备编号，默认使用全部 8 张卡。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# 每节点训练进程数，默认与 8 卡 FSDP 配置一致。
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# torchrun 主进程通信端口。
export MASTER_PORT="${MASTER_PORT:-50012}"
# 禁止 Hugging Face Hub 联网下载。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# 禁止 Transformers 联网加载模型文件。
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# 禁止 Datasets 联网加载数据集。
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# 启用 CPU 视觉位置索引、Index Put 和 DeepStack Index Add 优化。
export COSMOS_QWEN3VL_VISUAL_INDEX="${COSMOS_QWEN3VL_VISUAL_INDEX:-1}"
# 在 CPU 预计算 Vision Attention 的累积序列长度，减少重复设备到主机同步。
export COSMOS_QWEN3VL_VISION_HOST_CU_SEQLENS="${COSMOS_QWEN3VL_VISION_HOST_CU_SEQLENS:-1}"
# 使用 Vision Eager RoPE 路径，保留原有 Text RoPE 路径。
export COSMOS_QWEN3VL_VISION_ROPE_EAGER="${COSMOS_QWEN3VL_VISION_ROPE_EAGER:-1}"
# 启用 Sparse Chunked CE，默认每块 4096 个监督 Token，0 表示关闭。
export COSMOS_TORCH_CHUNKED_CE_TOKENS="${COSMOS_TORCH_CHUNKED_CE_TOKENS:-4096}"
# 首次 Chunked CE 调用时记录监督 Token 数及分块大小。
export COSMOS_CHUNKED_CE_LOG_SHAPES="${COSMOS_CHUNKED_CE_LOG_SHAPES:-1}"

# 启用 RMSNorm 快速实现，减少反向计算开销。
export USE_FAST_RMS_LAYER_NORM="${USE_FAST_RMS_LAYER_NORM:-1}"
# 设置 BKCL 最大 Cluster 数，使用已验证的 8 Cluster 配置。
export BKCL_MAX_CLUSTERS="${BKCL_MAX_CLUSTERS:-8}"
# 启用 XDNN Fast GeLU 实现。
export XDNN_USE_FAST_GELU="${XDNN_USE_FAST_GELU:-1}"
# 清除旧 Fast Swish 开关，避免与 Faster Swish 路径冲突。
unset XDNN_USE_FAST_SWISH
# 启用 XDNN Faster Swish（SiLU）实现。
export XDNN_USE_FASTER_SWISH="${XDNN_USE_FASTER_SWISH:-true}"
# XFA Attention 内部采用 FP32 累加。
export XFA_ACCUM_TYPE="${XFA_ACCUM_TYPE:-float}"
# XFA Attention 内部 GEMM 采用 FP16 计算。
export XFA_GEMM_TYPE="${XFA_GEMM_TYPE:-float16}"
# 设置 XBLAS GEMM 计算类型为 BF16，与长程实验一致。
export XBLAS_GEMM_COMPUTE_TYPE="${XBLAS_GEMM_COMPUTE_TYPE:-bf16}"
# 启用 XPU Runtime 的 Greedy Init 配置，与长程实验一致。
export XPUCUDA_GREEDY_INIT="${XPUCUDA_GREEDY_INIT:-1}"
# 对显存分配大小进行二次幂分档取整，提高缓存复用率。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-${PYTORCH_ALLOC_CONF:-roundup_power2_divisions:4}}"
# 同步 XPU 2609 Runtime 读取的旧版分配器环境变量。
export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"

# 每 Rank 最大 Batch Size，默认使用已完成 500 步验证的 BS14。
export LLAVA_OV_MAX_BATCH_SIZE="${LLAVA_OV_MAX_BATCH_SIZE:-14}"
# Batch Token 预算，沿用长程性能实验的 131072。
export LLAVA_OV_MAX_TOKENS="${LLAVA_OV_MAX_TOKENS:-131072}"
# 动态组批样本池大小，默认使用已完成 500 步验证的 128。
export LLAVA_OV_POOL_SIZE="${LLAVA_OV_POOL_SIZE:-128}"
# 性能回调预热步数，沿用长程实验的 5。
export LLAVA_OV_PERF_HIT_THRES="${LLAVA_OV_PERF_HIT_THRES:-5}"
# 是否跳过最终 Checkpoint 保存，正常训练默认保留，纯性能实验可设为 1。
export COSMOS_SKIP_FINAL_CHECKPOINT="${COSMOS_SKIP_FINAL_CHECKPOINT:-0}"
# ================== 环境变量配置结束 ==================

TAIL_OVERRIDES=(
    "model.config.policy.backbone.model_name=$MODEL_PATH"
    # 与已验证的 BS14 / Pool128 动态组批及性能统计配置对齐；外部 EXTRA_TAIL_OVERRIDES 仍可覆盖。
    "dataloader_train.batcher.max_batch_size=$LLAVA_OV_MAX_BATCH_SIZE"
    "dataloader_train.batcher.max_tokens=$LLAVA_OV_MAX_TOKENS"
    "dataloader_train.batcher.pool_size=$LLAVA_OV_POOL_SIZE"
    "trainer.callbacks.iter_speed.hit_thres=$LLAVA_OV_PERF_HIT_THRES"
    "trainer.callbacks.vlm_tokens_per_sec.hit_thres=$LLAVA_OV_PERF_HIT_THRES"
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
