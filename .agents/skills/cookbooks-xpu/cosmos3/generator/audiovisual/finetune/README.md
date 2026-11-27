<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Vision Generator Fine-Tuning (SFT) — Kunlun P800

Use the existing P800 launchers and TOMLs under `examples/` for T2V / I2V / V2V
vision SFT. This directory contains usage documentation only. Run all commands
**from the cosmos-framework repository root**, with the
[XPU environment](../../../README.md#cosmos-framework-kunlun-branch) active.

| Recipe                    | P800 entry point                                                                                                                       | Base model    |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Nano vision SFT           | [launch_sft_vision_nano_p800.sh](../../../../../../../examples/launch_sft_vision_nano_p800.sh)                                         | Cosmos3-Nano  |
| Edge vision SFT           | [launch_sft_vision_edge_p800.sh](../../../../../../../examples/launch_sft_vision_edge_p800.sh)                                         | Cosmos3-Edge  |
| Nano map-style vision SFT | [launch_sft_vision_nano_mapstyle_dataloader_p800.sh](../../../../../../../examples/launch_sft_vision_nano_mapstyle_dataloader_p800.sh) | Cosmos3-Nano  |
| Super vision LoRA         | [launch_sft_vision_super_p800.sh](../../../../../../../examples/launch_sft_vision_super_p800.sh)                                       | Cosmos3-Super |

The launchers enable offline HF access and expect prepared local data, VAE,
processor/tokenizer and DCP weights. They do not stage those inputs automatically.
All vision recipes use the same Bridge captioned-video dataset and prefer
structured `caption_json` prompts.

## Prepare shared data and VAE

Accept the Hugging Face model/dataset terms and authenticate with
`uvx hf@latest auth login` or `HF_TOKEN` before downloading.

```shell
export DATASET_PATH="$PWD/examples/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge"
export WAN_VAE_PATH="$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"

uvx hf@latest download --repo-type dataset nvidia/BridgeData2-Subset-Synthetic-Captions \
    --revision 40d018ac1c1a2a4b9734f17fdb21f3d933c49a01 \
    --local-dir "$(dirname "$DATASET_PATH")"
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir "$(dirname "$WAN_VAE_PATH")"
```

The dataset root must contain `train/video_dataset_file.jsonl`.
Keep the container's P800 Torch/XPU stack and library paths.

## Nano and Nano map-style

Both use the Nano base DCP and Qwen3-VL-8B tokenizer metadata:

```shell
export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Nano"
export TOKENIZER_PATH="$PWD/examples/Qwen3-VL-8B-Instruct"

python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path Cosmos3-Nano -o "$BASE_CHECKPOINT_PATH"
uvx hf@latest download Qwen/Qwen3-VL-8B-Instruct \
    --include '*.json' '*.txt' '*.jinja' --local-dir "$TOKENIZER_PATH"

# Choose one launcher.
bash examples/launch_sft_vision_nano_p800.sh
bash examples/launch_sft_vision_nano_mapstyle_dataloader_p800.sh
```

The historical map-style name currently selects the `CosmosDataLoader` stack:
`RankPartitionedDistributor` → `IdentityProcessor` → `SequentialPackingBatcher`
→ `VFMListCollator`. Its `SFTDataset` is iterable; the name does not guarantee
exact data-position resume. The token budget belongs to the nested batcher,
not a top-level `dataloader_train.max_sequence_length` override. The map-style
wrapper also sets `LOG_FILTER_REGEX` to suppress benign text-truncation notices.

## Edge

Edge uses its native processor rather than the Qwen tokenizer. Keep the HF
snapshot and DCP in separate directories, matching the existing P800 launcher:

```shell
export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Edge-DCP"
export EDGE_PROCESSOR_PATH="$PWD/examples/checkpoints/Cosmos3-Edge"

uvx hf@latest download nvidia/Cosmos3-Edge --local-dir "$EDGE_PROCESSOR_PATH"
python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path Cosmos3-Edge -o "$BASE_CHECKPOINT_PATH"
bash examples/launch_sft_vision_edge_p800.sh
```

Use absolute `EDGE_PROCESSOR_PATH` and `WAN_VAE_PATH` values: the wrapper exports
`COSMOS_HF_LOCAL__nvidia__Cosmos3_Edge` and `COSMOS_HF_LOCAL__Wan_AI__Wan2_2_TI2V_5B`
before its common path resolver runs. Keep the native processor files intact.
The Edge generation recipe has no sound tokenizer.

## Super LoRA

The [Super TOML](../../../../../../../examples/toml/sft_config/vision_sft_super.toml)
defines generation-branch LoRA with rank 16, alpha 32, lr `5e-4`, EMA disabled,
and a Cosmos3-Super / Qwen3-VL-32B base. Keep the HF snapshot and converted DCP
in separate directories:

```shell
export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Super-DCP"
export SUPER_PROCESSOR_PATH="$PWD/examples/checkpoints/Cosmos3-Super"

uvx hf@latest download nvidia/Cosmos3-Super --local-dir "$SUPER_PROCESSOR_PATH"
python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path Cosmos3-Super -o "$BASE_CHECKPOINT_PATH"
bash examples/launch_sft_vision_super_p800.sh
```

The P800 Super wrapper overrides HSDP to shard 4 / replicate 2
(`data_parallel_shard_degree=4`, `data_parallel_replicate_degree=2`) and sets
`+model.config.parallelism.fsdp_frozen_param_dtype=bfloat16` for frozen base
weights alongside the standard P800 runtime switches.

## Runtime and parameter overrides

All four vision P800 wrappers select FlashAttention-2 (`COSMOS_ATTN_BACKENDS=flash2`),
P800 fused kernels, P800 VAE/memory switches (see
[P800 runtime switches](../../../README.md#p800-runtime-switches)), BF16 model
computation, FP16 VAE computation, and seed 42, with `torch.compile`,
`device_monitor`, and `ofu` callbacks disabled and `prefetch_factor=32`
(`enable_async_batch_building=true` on Nano, Edge, and Super).

| Variable                                                              | Purpose                                                                      |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `DATASET_PATH`                                                        | directory containing `train/video_dataset_file.jsonl`                        |
| `BASE_CHECKPOINT_PATH`                                                | matching Nano, Edge, or Super DCP                                            |
| `WAN_VAE_PATH`                                                        | Wan2.2 VAE file                                                              |
| `TOKENIZER_PATH`                                                      | Nano/map-style tokenizer metadata                                            |
| `EDGE_PROCESSOR_PATH`                                                 | native Edge processor snapshot                                               |
| `SUPER_PROCESSOR_PATH`                                                | local Cosmos3-Super processor snapshot                                       |
| `OUTPUT_ROOT`                                                         | log/run parent; defaults to each wrapper's timestamped path under `outputs/` |
| `IMAGINAIRE_OUTPUT_ROOT`                                              | explicit artifact root; defaults to `OUTPUT_ROOT`                            |
| `CUDA_VISIBLE_DEVICES`                                                | visible devices; eight P800s by default                                      |
| `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` | torchrun topology                                                            |
| `COSMOS_DETERMINISTIC`                                                | set to `1` to pass `--deterministic` via the common launcher                 |

The common example launcher resolves relative data/checkpoint paths against the
framework repository root. Prefer absolute tokenizer/processor paths.
The wrappers construct fixed `TAIL_OVERRIDES` arrays and do not consume
`EXTRA_TAIL_OVERRIDES`. For custom smoke-test iteration counts, use the
[raw training entry point](../../../../../../../docs/training.md#option-b-raw-torchrun)
with the selected P800 settings and append Hydra overrides after `--`.

## Outputs and checkpoint export

Each wrapper chooses a timestamped output root by default. Set `OUTPUT_ROOT`
explicitly when you want a stable job directory, for example:

```shell
OUTPUT_ROOT="$PWD/outputs/vision_nano_p800" \
    bash examples/launch_sft_vision_nano_p800.sh
```

For that Nano run, the resolved job directory is
`outputs/vision_nano_p800/cosmos3/sft/vision_sft_nano/`; console logs are under
`outputs/vision_nano_p800/logs/`. Keep the same output root and job identity for
a same-job restart. Exact data-order resume depends on the loader/callbacks.

```shell
RUN_DIR=outputs/vision_nano_p800/cosmos3/sft/vision_sft_nano
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

Vision SFT disables action generation; use base/action-trained checkpoints for
action tasks. See the [inference reference](../../../../../../../docs/inference.md),
[structured TOML reference](../../../../../../../docs/sft_config.md), and
[upstream audiovisual cookbook](../../../../../cookbooks/cosmos3/generator/audiovisual/finetune/README.md).
