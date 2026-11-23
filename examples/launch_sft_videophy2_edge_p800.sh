#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# ============================================================================
# Structured-TOML launch for videophy2_sft_edge on Kunlun P800 (8-GPU FSDP).
# Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/videophy2_sft_edge.toml.
#
# The generic (CUDA/GB200) counterpart is examples/launch_sft_videophy2_edge.sh;
# this file exists only to add the P800 environment and a fully-offline,
# local-path-only setup. The recipe TOML, the experiment Python and the data
# pipeline are deliberately left untouched.
#
# Fully offline: this launcher never touches the HF Hub, S3 or GCS. The Edge
# backbone is a LOCAL SNAPSHOT DIRECTORY (`nvidia/Cosmos3-Edge` downloaded
# verbatim) which supplies arch/config/tokenizer AND the reasoner weights — the
# training loader follows the snapshot's root `model.safetensors.index.json`
# into its shards, so there is no converter step and no separate weights env
# var (docs/training.md:157).
#
#   * model.config.policy.backbone.model_name -> local snapshot dir. Short-
#     circuited by maybe_download_hf_model_from_s3 (pretrained_models_downloader
#     `if os.path.isdir(...): return ...`), and consumed by the dataset's
#     processor node via the Hydra interpolation
#     `tokenizer_type="${model.config.policy.backbone.model_name}"` so
#     build_processor takes its local-dir branch and, for a native
#     `model_type=cosmos3_edge` snapshot, builds Nemotron3DenseVLProcessor
#     (processors/__init__.py:131-132).
#   * `backbone.safetensors_path` is left EMPTY on purpose: with it unset,
#     vlm_model.py:617-624 loads weights from the resolved model_name dir.
#
# Env vars (override for your filesystem):
#   VIDEOPHYSICS_ROOT     VideoPhy-2 data root; must contain videophy2_train/
#                         and videophy2_val/ (each with meta.json, media/, text/).
#   MODEL_PATH            local nvidia/Cosmos3-Edge snapshot dir (config +
#                         processor + tokenizer + safetensors index + shards).
#                         NOT a Hub repo id -- a repo id would require network.
#   OUTPUT_ROOT           training output root (default under repo output/).
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8).
#   MASTER_PORT           torchrun --master_port (default 50013).
#   MAX_ITER              trainer.max_iter (default 500).
#   SAVE_ITER             checkpoint.save_iter (default 500). NOTE the recipe
#                         default is 100 while max_iter is 50, so a stock run
#                         would save nothing and hf_export would never fire.
#   TRAIN_PRECISION       model.config.precision (default bfloat16).
#   ATTN_IMPLEMENTATION   model.config.policy.attn_implementation (default
#                         sdpa). Edge must NOT inherit the Qwen3-VL-specific
#                         "cosmos" adapter (vlm_policy.py:70-85); sdpa is the
#                         portable choice and needs no flash-attn wheel. Set
#                         flash_attention_2 only after it is verified on XPU.
#   FSDP_MASTER_DTYPE     model.config.parallelism.fsdp_master_dtype (default
#                         float32). Leave at float32: it is the optimizer-facing
#                         master dtype, and bf16 there rounds away this recipe's
#                         tiny 1e-6 LR steps (same argument as the nano launcher).
#   FSDP_DELAY_RS         defer FSDP reduce-scatter through gradient
#                         accumulation (default 1 on P800). Edge is only ~2.4B,
#                         so the retained full-gradient cost is ~5 GiB/rank --
#                         unlike the 32B Super tier, this stays safe.
#   AC_MODE               activation checkpointing: none | full | selective
#                         (default none). "full" is the recipe default and was
#                         measured 10.6% slower (12.18 vs 10.89 s/iter) with
#                         identical loss; "none" costs 43.5 -> 58.3 GiB peak
#                         memory of 96 GiB. "selective" = SAC, needs the regex.
#   SAC_SAVE_FLASH        legacy: 1 pins SAC_SAVE_OPS_REGEX and requires
#                         AC_MODE=selective. Default 0.
#   EDGE_SDPA_CAUSAL      mask-free SDPA (default 1, measured -12.9%). When the batch mask is
#                         right padding only, skip the additive [B,1,S,S] causal
#                         mask and let SDPA take is_causal=True: -204 MB at
#                         S=10096 and 2.1-2.4x faster attention on P800.
#   JOB_NAME              job.name (default videophy2_500step_edge). A fixed
#                         name means a re-run auto-resumes from that run dir's
#                         latest checkpoint -- pass a new name to start fresh.
#   WANDB_MODE            job.wandb_mode (default online_force_new_id). Set
#                         disabled for smoke runs. online_* requires
#                         WANDB_API_KEY in the environment.
#   RUN_VALIDATION        trainer.run_validation (default false).
#   CYCLE_LENGTHS         scheduler.cycle_lengths (default [500]). MUST cover
#                         MAX_ITER -- see the note on the override below.
#   VP2_LENGTH_BALANCE    length-balanced rank assignment (default 1 on P800).
#                         The shipped length index was calibrated with the Qwen
#                         tokenizer; for Edge the ABSOLUTE lengths differ but the
#                         resolution ordering is unchanged, so the bucketing
#                         property still holds. Set 0 for the stock order.
#   EXTRA_TAIL_OVERRIDES  extra space-separated Hydra overrides, appended last
#                         (so they win over every default above).
#
# IMPORTANT: do NOT clear LD_LIBRARY_PATH here. The repo AGENTS.md advice to run
# `export LD_LIBRARY_PATH=''` applies to NVIDIA NGC containers only; on Kunlun
# XPU it removes /opt/xre/so and the process dies with
# `ImportError: libxpurt.so.2: cannot open shared object file`.
#
# Usage (8-GPU allocation, from the repo root):
#   bash examples/launch_sft_videophy2_edge_p800.sh
#
# 1-process preflight (isolates model/processor/data before touching 8 cards):
#   NPROC_PER_NODE=1 MAX_ITER=1 SAVE_ITER=1 WANDB_MODE=disabled \
#   EXTRA_TAIL_OVERRIDES="job.name=edge_preflight" \
#   bash examples/launch_sft_videophy2_edge_p800.sh
#
# 2-step smoke (validates decode -> forward -> backward -> DCP -> HF export):
#   MAX_ITER=2 SAVE_ITER=2 WANDB_MODE=disabled \
#   EXTRA_TAIL_OVERRIDES="job.name=videophy2_smoke_edge_2step trainer.validation_iter=999999 trainer.max_val_iter=0" \
#   bash examples/launch_sft_videophy2_edge_p800.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/videophy2_sft_edge.toml"

# Both are required: the snapshot and the VideoPhy-2 tree are not in the repo.
: "${VIDEOPHYSICS_ROOT:?set VIDEOPHYSICS_ROOT to the dir holding videophy2_train/ and videophy2_val/}"
: "${MODEL_PATH:?set MODEL_PATH to a local nvidia/Cosmos3-Edge snapshot dir}"
: "${OUTPUT_ROOT:=outputs/videophy2_sft_edge_p800}"
: "${NPROC_PER_NODE:=8}"
: "${MAX_ITER:=500}"
: "${SAVE_ITER:=500}"
: "${TRAIN_PRECISION:=bfloat16}"
# Keep the optimizer-facing master copy in float32: this recipe trains at
# lr=1e-6, and bf16's ~8 mantissa bits round a 1e-6 step to zero at |w|~1e-2.
: "${FSDP_MASTER_DTYPE:=float32}"
# Deferring reduce-scatter through GA keeps the UNSHARDED gradient resident.
# Edge is ~2.4B -> ~5 GiB/rank, which is well inside the 96 GiB budget.
: "${FSDP_DELAY_RS:=1}"
if [[ "$FSDP_DELAY_RS" != "0" && "$FSDP_DELAY_RS" != "1" ]]; then
    echo "ERROR: FSDP_DELAY_RS must be 0 or 1" >&2
    exit 1
fi
export FSDP_DELAY_RS
# Default 1: let the SDPA backend use `is_causal=True` instead of the additive
# [B,1,S,S] causal mask `_update_causal_mask` builds. Only taken when the
# incoming 2D mask is right padding only, where the two are the same mask.
# Drops a 204 MB mask at S=10096; measured on 8 cards 17.11 -> 14.89 s/step
# (-12.9%) with the loss trajectory converging to the identical values by
# iteration 8. Set to 0 to fall back to the masked path.
: "${EDGE_SDPA_CAUSAL:=1}"
if [[ "$EDGE_SDPA_CAUSAL" != "0" && "$EDGE_SDPA_CAUSAL" != "1" ]]; then
    echo "ERROR: EDGE_SDPA_CAUSAL must be 0 or 1" >&2
    exit 1
fi
export EDGE_SDPA_CAUSAL

# Edge delta #1: attention backend. "cosmos" is Qwen3-VL-only and rejects the
# Edge reasoner's explicit mask, so the recipe pins flash_attention_2 and falls
# back to sdpa where no flash-attn wheel exists (vlm_policy.py:74-85). On P800
# we pin sdpa explicitly: it is the path already exercised by the nano recipe's
# torch_sdpa fallback and needs no XPU flash-attn build.
: "${ATTN_IMPLEMENTATION:=sdpa}"
case "$ATTN_IMPLEMENTATION" in
    sdpa|flash_attention_2|eager) ;;
    *) echo "ERROR: ATTN_IMPLEMENTATION must be sdpa | flash_attention_2 | eager" >&2; exit 1 ;;
esac

# Run identity / logging / LR-cycle shape.
: "${JOB_NAME:=videophy2_500step_edge}"
: "${WANDB_MODE:=online_force_new_id}"
: "${RUN_VALIDATION:=false}"
: "${CYCLE_LENGTHS:=[500]}"

# ---------------------------------------------------------------------------
# Kunlun P800 environment (kept identical to the other P800 launchers)
# ---------------------------------------------------------------------------
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

# _sft_launcher_common.sh defaults to 50012, which on a shared node is routinely
# held by another job -- rendezvous then dies with EADDRINUSE. Pin a port here.
: "${MASTER_PORT:=50013}"
export MASTER_PORT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_TRAINING=1
export PYTHONHASHSEED=42

export XDNN_USE_FAST_SWISH=1
export XDNN_USE_FAST_GELU=1

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDART_DUMMY_REGISTER=1
unset CUDA_LAUNCH_BLOCKING
export XMLIR_ENABLE_LINEAR_FC_FUSION=1
export XMLIR_ENABLE_FAST_FC=1
export XPYTORCH_RUN_ENHANCE=1
# XPYTORCH_RUN_ENHANCE auto-selects CUDA_DEVICE_ORDER=OAM_ID in 8-card runs,
# which renumbers devices out from under CUDA_VISIBLE_DEVICES. Pin the bus
# order the other P800 launchers use so "8-15" really means bus 0x94..0xE7.
# : "${CUDA_DEVICE_ORDER:=PCI_BUS_ID}"
# export CUDA_DEVICE_ORDER
# Same XCCL topology hint as the qwen3.8-27b P800 production config.
export BKCL_PCIE_TOPO=1
# Keep FSDP units unsharded after forward: skips the backward re-all-gather and
# halves all-gather volume (all_gather was 15.6% of device kernel time). Costs one
# unsharded copy of the parameters per rank (~4.8 GB bf16 for Edge). Math-neutral.
: "${FSDP_NOSHARD_ALL:=0}"
export FSDP_NOSHARD_ALL
export XDNN_FAST_DIV_SCALAR=true
export XPUAPI_SDNN_BF16_ROUND_MODE=3

# These fused kernels are wired into the qwen3_vl generator blocks. The Edge
# reasoner uses its own Cosmos3EdgeRMSNorm (plain torch) and does not call them;
# they are kept so the P800 environment block stays identical across launchers.
export COSMOS_FUSED_RMSNORM=1
export COSMOS_FUSED_ROPE=1
export COSMOS_FUSED_CAUSAL_CONV_PAD=1
export COSMOS_FUSED_VAE_NORM=1

# The experiment Python reads this (data_sources_videophy2/videophy2.py:21).
export VIDEOPHYSICS_ROOT
export WANDB_PROJECT="${WANDB_PROJECT:-cosmos3-videophy2_edge}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
elif [[ "$WANDB_MODE" != "disabled" ]]; then
    echo "WARNING: WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is unset;" >&2
    echo "         export WANDB_API_KEY=... or run with WANDB_MODE=disabled." >&2
fi

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
_ct=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"model_type\",\"\"))" "$MODEL_PATH/config.json" 2>/dev/null || true)
[[ "$_ct" == "cosmos3_edge" ]] || {
    echo "ERROR: $MODEL_PATH is not a native Cosmos3-Edge snapshot (model_type=$_ct, want cosmos3_edge)." >&2
    echo "       Old nemotron_siglip2 snapshots use remote-code processing and are NOT supported by this recipe." >&2
    exit 1
}
for _f in chat_template.jinja preprocessor_config.json video_preprocessor_config.json tokenizer.json model.safetensors.index.json; do
    [[ -f "$MODEL_PATH/$_f" ]] || { echo "ERROR: snapshot missing $_f under $MODEL_PATH" >&2; exit 1; }
done
echo ">>> offline artifacts OK: data=$VIDEOPHYSICS_ROOT model=$MODEL_PATH (cosmos3_edge)"
'

TAIL_OVERRIDES=(
    # --- offline: local directory only, no Hub / S3 / GCS lookup ---
    "model.config.policy.backbone.model_name=$MODEL_PATH"
    # --- start from the pretrained snapshot weights ("" == load pretrain) ---
    "checkpoint.load_path="
    # --- Edge delta: never inherit the Qwen3-VL "cosmos" adapter ---
    "model.config.policy.attn_implementation=$ATTN_IMPLEMENTATION"
    # --- precision, matching the other P800 launchers ---
    "model.config.precision=$TRAIN_PRECISION"
    "model.config.parallelism.fsdp_master_dtype=$FSDP_MASTER_DTYPE"
    # --- topology: shard * replicate * cp * cfgp must equal WORLD_SIZE ---
    "model.config.parallelism.data_parallel_shard_degree=$NPROC_PER_NODE"
    "model.config.parallelism.data_parallel_replicate_degree=1"
    "model.config.parallelism.context_parallel_shard_degree=1"
    "model.config.parallelism.cfg_parallel_shard_degree=1"
    # --- first pass on P800: no torch.compile ---
    "model.config.compile.enabled=false"
    # --- reproducibility ---
    "trainer.seed=42"
    # --- run length / checkpoint cadence ---
    "trainer.max_iter=$MAX_ITER"
    "checkpoint.save_iter=$SAVE_ITER"
    # DCP async save stages the state dict through PINNED host memory and calls
    # cudaHostRegister, which is unavailable on P800 ("Registering memory failed
    # with cudaError: 2"). The supported fallback is the synchronous save path.
    "checkpoint.dcp_async_mode_enabled=false"
    # --- run identity / logging ---
    "job.name=$JOB_NAME"
    "job.wandb_mode=$WANDB_MODE"
    # --- validation off by default ---
    "trainer.run_validation=$RUN_VALIDATION"
    # --- LR cycle length, which MUST track max_iter: find_in_interval() returns
    #     None past sum(cycle_lengths) and schedule() then indexes with None.
    "scheduler.cycle_lengths=$CYCLE_LENGTHS"
)

# Activation checkpointing. The recipe default is "full" (checkpoint every
# transformer block, so backward replays a whole extra forward). Measured on
# 8xP800, same card set and window: full 12.18 s/iter -> none 10.89 s/iter
# (-10.6%), with the loss identical step by step; peak memory 43.5 -> 58.3 GiB
# of 96 GiB, i.e. there is headroom. So "none" is the default here.
#
#   AC_MODE=none       no checkpointing (default, fastest)
#   AC_MODE=full       checkpoint every block (recipe default; -10.6% slower)
#   AC_MODE=selective  per-op SAC. Needs save_ops_regex pinned to the P800 op
#                      names -- the nano recipe's `[_flash_attn.*forward]`
#                      targets flash2 dispatcher names and matches nothing under
#                      SDPA, which silently degrades to full recompute.
: "${AC_MODE:=none}"
case "$AC_MODE" in
    none|full|selective) ;;
    *) echo "ERROR: AC_MODE must be none | full | selective" >&2; exit 1 ;;
esac
if [[ "$AC_MODE" != "full" ]]; then
    TAIL_OVERRIDES+=("model.config.activation_checkpointing.mode=$AC_MODE")
fi

# Legacy knob kept so existing command lines keep working; it only pins the
# regex and requires AC_MODE=selective.
: "${SAC_SAVE_FLASH:=0}"
if [[ "$SAC_SAVE_FLASH" != "0" && "$SAC_SAVE_FLASH" != "1" ]]; then
    echo "ERROR: SAC_SAVE_FLASH must be 0 or 1" >&2
    exit 1
fi
if [[ "$SAC_SAVE_FLASH" == "1" ]]; then
    if [[ "$AC_MODE" != "selective" ]]; then
        echo "ERROR: SAC_SAVE_FLASH=1 requires AC_MODE=selective" >&2
        exit 1
    fi
    : "${SAC_SAVE_OPS_REGEX:=[_scaled_dot_product_flash_attention.default]}"
    TAIL_OVERRIDES+=(
        "model.config.activation_checkpointing.save_ops_regex=$SAC_SAVE_OPS_REGEX"
    )
fi

# Length-balanced rank assignment. Same mechanism as the nano launcher; the
# shipped length index used the Qwen tokenizer, so Edge's absolute lengths
# differ but the per-resolution ordering (and therefore the bucketing) holds.
: "${VP2_LENGTH_BALANCE:=1}"
if [[ "$VP2_LENGTH_BALANCE" != "0" && "$VP2_LENGTH_BALANCE" != "1" ]]; then
    echo "ERROR: VP2_LENGTH_BALANCE must be 0 or 1" >&2
    exit 1
fi
export VP2_LENGTH_BALANCE
if [[ "$VP2_LENGTH_BALANCE" == "1" && ! -f "$VIDEOPHYSICS_ROOT/videophy2_train/length_index.json" ]]; then
    echo "WARNING: length index missing at $VIDEOPHYSICS_ROOT/videophy2_train/length_index.json" >&2
    echo "         length balancing will fall back to the stock order (no speedup)." >&2
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

# ---------------------------------------------------------------------------
# Collective health gate. A degraded XLink/C2C link still answers a 1 KB
# all_reduce, so correctness is not a gate: require CC_MIN_BW GB/s of ring
# bandwidth and report any new KL_XID_C2C_EVENT in dmesg. Adopted from the
# qwen3.8-27b P800 runbook (/home/qwen/preflight_cc.sh); a link that has fallen
# back to PCIe runs at ~18.7 GB/s instead of ~48 GB/s and kills training within
# tens of steps.
# ---------------------------------------------------------------------------
: "${CC_PREFLIGHT:=1}"
: "${CC_MIN_BW:=40}"
# Single-card runs have no collective to measure (the ring bandwidth formula is 0
# for world size 1), so the gate only applies to real multi-card launches.
if [[ "$CC_PREFLIGHT" == "1" && "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
    _cc_log="$OUTPUT_ROOT/logs/cc_preflight.log"
    mkdir -p "$(dirname "$_cc_log")"
    _cc_before=$(dmesg 2>/dev/null | grep -c KL_XID_C2C_EVENT || true)
    echo ">>> collective preflight on ${CUDA_VISIBLE_DEVICES} (min ${CC_MIN_BW} GB/s)..."
    if ! timeout 420 torchrun --nproc_per_node="$NPROC_PER_NODE" \
            --master_port="$((MASTER_PORT + 7))" \
            "$(dirname "${BASH_SOURCE[0]}")/../tools/p800_cc_check.py" > "$_cc_log" 2>&1; then
        echo "ERROR: collective preflight did not complete; see $_cc_log" >&2
        exit 1
    fi
    _cc_bw=$(grep -oE "ALLREDUCE [0-9.]+" "$_cc_log" | head -1 | awk '{print $2}')
    _cc_after=$(dmesg 2>/dev/null | grep -c KL_XID_C2C_EVENT || true)
    _cc_new=$((_cc_after - _cc_before))
    echo ">>> collective preflight: all_reduce ${_cc_bw:-?} GB/s, new c2c events: $_cc_new"
    if [[ -z "${_cc_bw:-}" ]]; then
        echo "ERROR: preflight produced no bandwidth number; see $_cc_log" >&2
        exit 1
    fi
    if awk "BEGIN{exit !($_cc_bw < $CC_MIN_BW)}"; then
        echo "ERROR: collective bandwidth ${_cc_bw} GB/s < ${CC_MIN_BW} GB/s -- XLink degraded" >&2
        echo "       recover with 'xpu-smi -r -i <id>' per card or a driver reinstall, then re-run." >&2
        exit 1
    fi
    if [[ "$_cc_new" -gt 0 ]]; then
        echo "WARNING: $_cc_new new KL_XID_C2C_EVENT lines during preflight (link may be marginal)" >&2
    fi
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"