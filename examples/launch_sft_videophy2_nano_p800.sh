#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for videophy2_sft_nano on Kunlun P800 (8-GPU FSDP).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/videophy2_sft_nano.toml.
#
# The generic (CUDA/GB200) counterpart is examples/launch_sft_videophy2_nano.sh;
# this file exists only to add the P800 environment and a fully-offline,
# local-path-only setup. The recipe TOML, the experiment Python and the data
# pipeline are deliberately left untouched.
#
# Fully offline: this launcher never touches the HF Hub, S3 or GCS. Every model
# artifact is a LOCAL DIRECTORY, which is what makes that work:
#   * model.config.policy.backbone.model_name     -> local dir. Short-circuited by
#     maybe_download_hf_model_from_s3 (pretrained_models_downloader.py:222
#     `if os.path.isdir(...): return ...`), and consumed by the dataset's
#     processor node via the Hydra interpolation
#     `tokenizer_type="${model.config.policy.backbone.model_name}"`
#     (videophy2_sft_nano.py:116) so build_processor takes its local-dir branch
#     (processors/__init__.py:122).
#   * model.config.policy.backbone.safetensors_path -> local dir of the merged
#     Cosmos3-Nano LM + Qwen3-VL vision tower weights, as produced by
#     cosmos_framework.scripts.convert_model_to_vlm_safetensors.
#
# Env vars (override for your filesystem):
#   VIDEOPHYSICS_ROOT     VideoPhy-2 data root; must contain videophy2_train/
#                         and videophy2_val/ (each with meta.json, media/, text/).
#   MODEL_PATH            local Qwen3-VL-8B-Instruct HF dir (config + processor
#                         + tokenizer). NOT a Hub repo id -- a repo id would
#                         require network access.
#   VLM_SAFETENSORS_PATH  local dir of the merged Cosmos3-Nano-VLM safetensors.
#   OUTPUT_ROOT           training output root (default under repo output/).
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8).
#   MASTER_PORT           torchrun --master_port (default 50013).
#   MAX_ITER              trainer.max_iter (default 500).
#   SAVE_ITER             checkpoint.save_iter (default 500). NOTE the recipe
#                         default is 100 while max_iter is 50, so a stock run
#                         would save nothing and hf_export would never fire.
#   TRAIN_PRECISION       model.config.precision (default bfloat16).
#   FSDP_MASTER_DTYPE     model.config.parallelism.fsdp_master_dtype (default
#                         float32). Leave at float32: it is the optimizer-facing
#                         master dtype, and bf16 there rounds away this recipe's
#                         tiny 1e-6 LR steps (see the note at the definition).
#   FSDP_DELAY_RS         defer FSDP reduce-scatter through gradient
#                         accumulation (default 1 on P800; set 0 to restore
#                         the original per-microstep reduce-scatter behavior).
#   JOB_NAME              job.name (default videophy2_500step_noval). A fixed
#                         name means a re-run auto-resumes from that run dir's
#                         latest checkpoint -- pass a new name to start fresh.
#   WANDB_MODE            job.wandb_mode (default online_force_new_id, i.e. log
#                         to W&B online with a fresh run id each launch).
#   RUN_VALIDATION        trainer.run_validation (default false).
#   CYCLE_LENGTHS         scheduler.cycle_lengths (default [500]). MUST cover
#                         MAX_ITER -- see the note on the override below.
#   I4_ATTN_BACKENDS      cosmos attention backend allow-list (default
#                         torch_sdpa). flash2 also works on this path as of
#                         2026-09-24 and is ~1% faster; see the note at the
#                         definition for why the default did not change.
#   SAC_SAVE_FLASH        selective activation checkpointing: save flash-attn
#                         forward outputs and recompute the rest (default 1 on
#                         P800). Set 0 for the legacy full-block recompute. See
#                         the note at the definition for the regex detail.
#   VP2_LENGTH_BALANCE    length-balanced rank assignment (default 1 on P800).
#                         Reorders samples WITHIN each optimizer step so all
#                         ranks hold similar-length samples in a microstep; this
#                         attacks the rank-arrival skew that dominates the
#                         all_gather wait. Set 0 for the stock order. See the
#                         note at the definition for the evidence.
#   EXTRA_TAIL_OVERRIDES  extra space-separated Hydra overrides, appended last
#                         (so they win over every default above).
#
# IMPORTANT: do NOT clear LD_LIBRARY_PATH here. The repo AGENTS.md advice to run
# `export LD_LIBRARY_PATH=''` applies to NVIDIA NGC containers only; on Kunlun
# XPU it removes /opt/xre/so and the process dies with
# `ImportError: libxpurt.so.2: cannot open shared object file`.
#
# Usage (8-GPU allocation, from the repo root):
#   bash examples/launch_sft_videophy2_nano_p800.sh
#
# 2-step smoke (validates decode -> forward -> backward -> DCP -> HF export):
#   MAX_ITER=2 SAVE_ITER=2 \
#   EXTRA_TAIL_OVERRIDES="job.name=videophy2_smoke_2step trainer.validation_iter=999999 trainer.max_val_iter=0" \
#   bash examples/launch_sft_videophy2_nano_p800.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/videophy2_sft_nano.toml"

: "${VIDEOPHYSICS_ROOT:=/home/cosmos3/examples/data/videophysics}"
: "${MODEL_PATH:=/home/cosmos3/Qwen3-VL-8B-Instruct}"
: "${VLM_SAFETENSORS_PATH:=/home/cosmos3/Cosmos3-Nano-VLM}"
: "${OUTPUT_ROOT:=/home/cosmos3/cosmos-framework/output/videophy2_sft_nano_p800}"
: "${NPROC_PER_NODE:=8}"
: "${MAX_ITER:=500}"
: "${SAVE_ITER:=500}"
: "${TRAIN_PRECISION:=bfloat16}"
# FSDP master (optimizer-facing) parameter dtype. Keep this at float32 -- the
# framework default -- and do NOT copy `bfloat16` from the VFM P800 launchers.
# Per cosmos_framework/configs/base/defaults/parallelism.py:60-70, this is the
# dtype the optimizer READS AND WRITES, so setting it equal to `precision` opts
# out of mixed precision entirely. The reason it is fatal here but harmless in
# the vision recipe is the learning rate: bf16 has ~8 mantissa bits, so at
# |w|~1e-2 the smallest representable increment (ulp) is ~3e-5, while the
# optimizer step is ~lr.
#   vision_sft_nano  lr=1e-4 -> ~3.3 ulp  -> updates land
#   this recipe      lr=1e-6 -> ~0.03 ulp -> updates round to zero
# Measured on a bf16 tensor with a 1e-6 step: only 2.4% of weights at |w|=1e-2
# change at all, and 0% at |w|>=1e-1. That is what a flat loss looks like.
: "${FSDP_MASTER_DTYPE:=float32}"
# The validated P800 optimization: GA8 accumulates full gradients locally and
# performs FSDP reduce-scatter only on the final microstep of each optimizer
# update. This raises peak memory; override to 0 for the legacy behavior.
: "${FSDP_DELAY_RS:=1}"
if [[ "$FSDP_DELAY_RS" != "0" && "$FSDP_DELAY_RS" != "1" ]]; then
    echo "ERROR: FSDP_DELAY_RS must be 0 or 1" >&2
    exit 1
fi
export FSDP_DELAY_RS
# Opt-in LM-only FSDP experiments; defaults preserve the original behavior.
# Depth 0 keeps implicit prefetch; 2/3 explicitly prefetch the next units in
# forward and the preceding units in backward. Unit size counts LM layers,
# not ranks. Both can increase peak memory despite keeping shard_degree=8.
: "${FSDP_PREFETCH_DEPTH:=0}"
: "${FSDP_LM_UNIT_SIZE:=1}"
if [[ ! "$FSDP_PREFETCH_DEPTH" =~ ^[0-9]+$ || ! "$FSDP_LM_UNIT_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: FSDP_PREFETCH_DEPTH must be nonnegative and FSDP_LM_UNIT_SIZE positive integers" >&2
    exit 1
fi
export FSDP_PREFETCH_DEPTH FSDP_LM_UNIT_SIZE

# Run identity / logging / LR-cycle shape. The defaults reproduce the completed
# 500-step run; each remains overridable from the shell.
: "${JOB_NAME:=videophy2_500step_noval}"
: "${WANDB_MODE:=online_force_new_id}"
: "${RUN_VALIDATION:=false}"
: "${CYCLE_LENGTHS:=[500]}"

# ---------------------------------------------------------------------------
# Kunlun P800 environment (kept identical to the other P800 launchers)
# ---------------------------------------------------------------------------
# Card set. Override before invoking to move off a busy allocation:
#   CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 bash examples/launch_sft_videophy2_nano_p800.sh
# Keep the count in sync with NPROC_PER_NODE -- data_parallel_shard_degree is
# derived from it and must equal WORLD_SIZE.
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

# _sft_launcher_common.sh defaults to 50012, which on a shared node is routinely
# held by another job -- rendezvous then dies with
# `DistNetworkError: ... port: 50012 ... EADDRINUSE`. Pin a port here.
: "${MASTER_PORT:=50013}"
export MASTER_PORT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_TRAINING=1
# Attention backend for the cosmos attention frontend, as an ALLOW-LIST
# (model/attention/utils/environment.py:55).
#
# P800 reports capability (8,6) -> arch_tag 86, whose default candidate list is
# [flash2, cudnn, natten, torch_sdpa].
#
# flash2 is selectable on this path only once hf_attention_cosmos requests
# CausalType.DontCare instead of TopLeft (tracked as a separate change). The
# backends split on mask convention -- flash2/flash3 take BottomRight,
# cudnn/natten take TopLeft, torch_sdpa ignores it -- and DontCare is the only
# value all four accept, which is valid here because the adapter already
# guarantees q_len == kv_len. With the hard-coded TopLeft, flash2 is unselectable.
#
# The default stays torch_sdpa on throughput grounds alone: a 10-step A/B on the
# same cards, seed and data measured flash2 at 37.49 s/iter against torch_sdpa's
# 37.88 s/iter (~1%), which is not worth changing a default for. To opt in:
#   I4_ATTN_BACKENDS=flash2 bash examples/launch_sft_videophy2_nano_p800.sh
#
# Because this variable is an allow-list, naming a backend collapses the
# candidate list to exactly that backend, which makes a wrong pin fatal rather
# than harmless ("Could not find a compatible Attention backend..."). That is
# useful when verifying a backend change: a strict `flash2` fails loudly if any
# call site is incompatible, whereas `flash2,torch_sdpa` would fall back
# silently. Leaving it UNSET also works, but then the kernel varies per call site
# with the platform's candidate order, so it is pinned here instead.
: "${I4_ATTN_BACKENDS:=flash2}"
export I4_ATTN_BACKENDS
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

export COSMOS_FUSED_RMSNORM=1
export COSMOS_FUSED_ROPE=1
export COSMOS_FUSED_CAUSAL_CONV_PAD=1
export COSMOS_FUSED_VAE_NORM=1

# The experiment Python reads this (data_sources_videophy2/videophy2.py:21).
export VIDEOPHYSICS_ROOT
export VLM_SAFETENSORS_PATH
export WANDB_API_KEY="f82ccaaaab462d8d779429028d12bfd7bf0c1d19"
export WANDB_PROJECT="cosmos3-videophy2_nano"

# ---------------------------------------------------------------------------
# Pre-flight checks: fail before torchrun starts, with a path-level message
# instead of a training-time traceback.
# ---------------------------------------------------------------------------
EXTRA_DATASET_CHECK='
for _split in videophy2_train videophy2_val; do
    [[ -f "$VIDEOPHYSICS_ROOT/$_split/meta.json" ]] || {
        echo "ERROR: missing $VIDEOPHYSICS_ROOT/$_split/meta.json" >&2
        echo "       run: python -m cosmos_framework.scripts.reasoner.prepare_videophy2_from_hf --out_root \$VIDEOPHYSICS_ROOT --split both" >&2
        exit 1
    }
done
[[ -d "$MODEL_PATH" ]] || { echo "ERROR: MODEL_PATH is not a directory: $MODEL_PATH" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "ERROR: MODEL_PATH has no config.json: $MODEL_PATH" >&2; exit 1; }
[[ -d "$VLM_SAFETENSORS_PATH" ]] || { echo "ERROR: VLM_SAFETENSORS_PATH is not a directory: $VLM_SAFETENSORS_PATH" >&2; exit 1; }
[[ -f "$VLM_SAFETENSORS_PATH/model.safetensors.index.json" ]] || {
    echo "ERROR: VLM_SAFETENSORS_PATH is not a merged VLM dir: $VLM_SAFETENSORS_PATH" >&2
    echo "       build it with: python -m cosmos_framework.scripts.convert_model_to_vlm_safetensors" >&2
    exit 1
}
echo ">>> offline artifacts OK: data=$VIDEOPHYSICS_ROOT model=$MODEL_PATH weights=$VLM_SAFETENSORS_PATH"
'

TAIL_OVERRIDES=(
    # --- offline: local directories only, no Hub / S3 / GCS lookup ---
    "model.config.policy.backbone.model_name=$MODEL_PATH"
    "model.config.policy.backbone.safetensors_path=$VLM_SAFETENSORS_PATH"
    # --- start from the pretrained merged weights ("" == load pretrain) ---
    "checkpoint.load_path="
    # --- precision, matching the other P800 launchers ---
    "model.config.precision=$TRAIN_PRECISION"
    # NOTE: model.config.tokenizer.dtype is NOT settable here. That node exists
    # only in the VFM config tree (it is the Wan VAE dtype used by the vision /
    # DROID recipes). This recipe runs the reasoner tree and instantiates no VAE,
    # so overriding it aborts with
    #   ConfigCompositionException: Could not override 'model.config.tokenizer.dtype'.
    # Removed 2026-10-12; re-add only if this recipe ever gains a VAE node.
    "model.config.parallelism.fsdp_master_dtype=$FSDP_MASTER_DTYPE"
    # --- topology: shard * replicate * cp * cfgp must equal WORLD_SIZE ---
    "model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE"
    "model.config.parallelism.data_parallel_replicate_degree=1"
    "model.config.parallelism.context_parallel_shard_degree=1"
    "model.config.parallelism.cfg_parallel_shard_degree=1"
    # --- first pass on P800: no torch.compile. (No tokenizer-compile key here:
    #     trainer.callbacks.compile_tokenizer exists only in the VFM config tree,
    #     not the reasoner/vlm one; the recipe TOML already sets it to false.) ---
    "model.config.compile.enabled=false"
    # --- reproducibility ---
    # NOTE: trainer.callbacks.device_monitor / .ofu / .compile_tokenizer are NOT
    # overridden here: those nodes exist only in the VFM config tree. This recipe
    # runs the reasoner tree (callbacks = basic_vlm, basic_log, hf_export).
    "trainer.seed=42"
    # --- run length / checkpoint cadence ---
    "trainer.max_iter=$MAX_ITER"
    "checkpoint.save_iter=$SAVE_ITER"
    # DCP async save stages the state dict through PINNED host memory
    # (checkpoint/dcp.py:_checkpoint_async_with_pinned_memory ->
    # torch/cuda/_pin_memory_utils.pin_memory -> cudart.cudaHostRegister).
    # cudaHostRegister is not available on this P800 platform: it fails with
    # `RuntimeError: Registering memory failed with cudaError: 2` and kills every
    # rank at the first save, after the HF export has already been written.
    # AsyncMode has only DISABLED / ASYNC_WITH_PINNED_MEM, so the supported
    # fallback is the synchronous save path. Cost: the save briefly blocks the
    # training loop instead of overlapping with it.
    "checkpoint.dcp_async_mode_enabled=false"
    # --- run identity / logging ---
    "job.name=$JOB_NAME"
    "job.wandb_mode=$WANDB_MODE"
    # --- validation off by default ---
    "trainer.run_validation=$RUN_VALIDATION"
    # --- LR cycle length, which MUST track max_iter ---
    # The recipe TOML pairs [trainer].max_iter = 50 with
    # [scheduler].cycle_lengths = [50]. find_in_interval()
    # (cosmos_framework/utils/functional/lr_scheduler.py:118) falls off the end of
    # its loop and returns None once the cumulative cycle boundary is passed, and
    # schedule() then indexes with that None:
    #   KeyValidationError: ListConfig indices must be integers or slices, not NoneType
    # So any max_iter above sum(cycle_lengths) aborts at step sum+1 -- with the
    # stock [50] that is exactly iteration 51. Keep the two in step.
    "scheduler.cycle_lengths=$CYCLE_LENGTHS"
)

# Selective activation checkpointing (SAC). The recipe default is full-block AC
# (model.config.activation_checkpointing.mode=full), which recomputes the whole
# transformer block in backward -- flash-attention included. SAC instead SAVES
# the flash-attention forward outputs and recomputes only the cheap ops, cutting
# the most expensive recompute for ~1.7 GB more peak memory. A 500-step run on
# P800 (videophy2_500step_sac_fp32_20261026) validated clean numerics, save and
# exit; the ~1.8% per-step speedup is from two reversed-order matched windows.
# Default on after that validation; set SAC_SAVE_FLASH=0 for legacy full-block AC.
#
# The regex MUST be [_flash_attn.*forward], NOT the schema default ["fmha"]:
# fmha does not match the P800 flash2 dispatcher op names, so a bare mode flip to
# selective would recompute attention anyway and show no benefit.
: "${SAC_SAVE_FLASH:=1}"
if [[ "$SAC_SAVE_FLASH" != "0" && "$SAC_SAVE_FLASH" != "1" ]]; then
    echo "ERROR: SAC_SAVE_FLASH must be 0 or 1" >&2
    exit 1
fi
if [[ "$SAC_SAVE_FLASH" == "1" ]]; then
    TAIL_OVERRIDES+=(
        "model.config.activation_checkpointing.mode=selective"
        "model.config.activation_checkpointing.save_ops_regex=[_flash_attn.*forward]"
    )
fi

# Length-balanced rank assignment. Sample lengths on this dataset fall into only
# three values (976 / 2672 / 5632 tokens, a 5.8x spread), so whichever rank holds
# a long sample in a given microstep arrives at all_gather last and the other
# seven wait. Measured on the current baseline: arrival skew median 17.29 ms vs
# only 5.49 ms of transfer after the last rank arrives. Reordering WITHIN each
# optimizer step puts similar lengths on all eight ranks per microstep.
#
# Validated by a formal 500-step run (lb500fix_20261029): median 33.890 -> 22.140
# s/step (-34.7%), flat across all ten 50-step buckets, peak memory unchanged
# (73.0 GiB, all eight cards). Correctness is the FP32 delay-RS tier: every
# step's sample set, per-rank sample count and per-sample weight are unchanged
# (proved offline in cosmos_framework/data/reasoner/length_balance.py and by
# output/videophy2_sft_nano_p800/verify_block_permutation.py), so only the
# floating-point accumulation order differs.
#
# The length index is metadata-only and built by
#   python -m tools.videophy2_length_index
# A missing index degrades to the stock order with a warning rather than failing.
#
# The block grid is anchored to the GLOBAL step index via epoch_block_offset(),
# which is required because 3343 % (world*ga=64) == 15. Anchoring it to each
# epoch's list instead let the grid drift 15 positions per epoch; that broke the
# set-invariance property at 448 of 500 steps and eroded the gain from 22.13 to
# 26.56 s/step. Keep the offset -- do not "simplify" it away.
: "${VP2_LENGTH_BALANCE:=1}"
if [[ "$VP2_LENGTH_BALANCE" != "0" && "$VP2_LENGTH_BALANCE" != "1" ]]; then
    echo "ERROR: VP2_LENGTH_BALANCE must be 0 or 1" >&2
    exit 1
fi
export VP2_LENGTH_BALANCE
if [[ "$VP2_LENGTH_BALANCE" == "1" && ! -f "$VIDEOPHYSICS_ROOT/videophy2_train/length_index.json" ]]; then
    echo "WARNING: length index missing at $VIDEOPHYSICS_ROOT/videophy2_train/length_index.json" >&2
    echo "         length balancing will fall back to the stock order (no speedup)." >&2
    echo "         build it with: python -m tools.videophy2_length_index" >&2
fi

# Optional extra overrides from the caller's shell (word-split on purpose).
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    # noglob around the unquoted expansion: overrides legitimately contain brackets
    # (e.g. scheduler.cycle_lengths=[500]), which would otherwise be treated as a
    # glob pattern during word splitting and could be rewritten by the shell.
    set -f
    # shellcheck disable=SC2206
    TAIL_OVERRIDES+=( ${EXTRA_TAIL_OVERRIDES} )
    set +f
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
