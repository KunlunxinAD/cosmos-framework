#!/usr/bin/env bash

TOML_FILE="examples/toml/sft_config/vision_sft_nano.toml"
: "${DATASET_PATH:=/workspace/cosmos-framework/checkpoint/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=/workspace/cosmos-framework/checkpoint/Cosmos3-Nano-DCP}"
: "${WAN_VAE_PATH:=/workspace/cosmos-framework/checkpoint/Wan2.2_VAE.pth}"
: "${TOKENIZER_PATH:=/cosmos3/Qwen3-VL-8B-Instruct}"
: "${TRAIN_PRECISION:=bfloat16}"
: "${VAE_DTYPE:=float16}"
: "${OUTPUT_ROOT:=outputs/vision_sft_nano_$(date +%Y%m%d_%H%M%S)}"
: "${NPROC_PER_NODE:=8}"
: "${MAX_ITER:=500}"
: "${FP16_COMPUTE_MLP:=1}"

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
    "model.config.compile.enabled=false"
    "model.config.fp16_compute_mlp=$FP16_COMPUTE_MLP"
    "trainer.seed=42"
    "trainer.callbacks.device_monitor.every_n=0"
    "trainer.callbacks.ofu.every_n=0"
    "trainer.max_iter=$MAX_ITER"
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"