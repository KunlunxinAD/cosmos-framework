<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano Action-Policy Fine-Tuning (SFT)— Kunlun P800

Use the P800 launchers under the framework's `examples/` directory for DROID
and LIBERO training. This cookbook documents input preparation and invocation;
scripts and TOMLs are maintained only in `examples/`.
Run every command below **from the cosmos-framework repository root** with the
[XPU environment](../../../README.md#cosmos-framework-kunlun-branch) active.

| Recipe | P800 launcher | Task | Dataset |
| --- | --- | --- | --- |
| DROID policy | [launch_sft_action_policy_droid_nano_p800.sh](../../../../../../../examples/launch_sft_action_policy_droid_nano_p800.sh) | `joint_pos` action policy | Cosmos3-DROID success |
| DROID forward dynamics | [launch_sft_action_fd_droid_posttrain_p800.sh](../../../../../../../examples/launch_sft_action_fd_droid_posttrain_p800.sh) | future visual prediction | Cosmos3-DROID success + failure |
| LIBERO-10 policy | [launch_sft_action_policy_libero_10_nano_p800.sh](../../../../../../../examples/launch_sft_action_policy_libero_10_nano_p800.sh) | action policy | `libero_10` |
| LIBERO-all policy | [launch_sft_action_policy_libero_all_nano_p800.sh](../../../../../../../examples/launch_sft_action_policy_libero_all_nano_p800.sh) | four-suite policy | all LIBERO suites |

The paired configurations are in
[examples/toml/sft_config/](../../../../../../../examples/toml/sft_config/).
The P800 scripts set FlashAttention-2, fused kernels, FP16 VAE computation and
single-node runtime settings. They enable offline HF access during training,
so prepare the inputs below before launching. They do not download the dataset,
VAE, tokenizer or base checkpoint automatically.

## Prepare shared inputs

Accept the required Hugging Face terms and authenticate with
`uvx hf@latest auth login` or `HF_TOKEN` first.

```shell
export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Nano"
export WAN_VAE_PATH="$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
export TOKENIZER_PATH="$PWD/examples/Qwen3-VL-8B-Instruct"

uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --local-dir "$(dirname "$WAN_VAE_PATH")"
uvx hf@latest download Qwen/Qwen3-VL-8B-Instruct \
    --include '*.json' '*.txt' '*.jinja' --local-dir "$TOKENIZER_PATH"
python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path Cosmos3-Nano -o "$BASE_CHECKPOINT_PATH"
```

Use a base checkpoint with action-head weights for these recipes. A vision-only
SFT export disables action generation and is not a replacement for that base.
Keep the P800 container's matching Torch/XPU stack and `LD_LIBRARY_PATH`.

## DROID

Stage [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID)
on storage with sufficient capacity. The merged dataset parent must be named
`droid_plus_lerobot_640x360_20260412`; policy training uses `success/`, while
forward dynamics uses both `success/` and `failure/`.

```shell
export DATASET_PATH="$PWD/examples/data/Cosmos3-DROID/droid_plus_lerobot_640x360_20260412"
uvx hf@latest download nvidia/Cosmos3-DROID --repo-type dataset \
    --local-dir "$DATASET_PATH"
export DROID_ROOT="$DATASET_PATH"

# Choose one run.
bash examples/launch_sft_action_policy_droid_nano_p800.sh
bash examples/launch_sft_action_fd_droid_posttrain_p800.sh
```

The policy wrapper derives `DROID_ROOT` from `DATASET_PATH` unless explicitly
set. Keep both consistent when relocating the dataset. The optional keep-ranges
filter is disabled in the registered policy experiment by default; these P800
wrappers do not download or configure it through a `FILTER_PATH` variable.
See [DROID policy training](../../../../../../../docs/action_policy_droid_posttrain.md)
and [forward-dynamics training](../../../../../../../docs/action_fd_droid_posttrain.md)
for task details.

## LIBERO

Use the 20 FPS
[LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3)
data. Download all suites once, or restrict the download to `libero_10/**`
when only running LIBERO-10.

```shell
uvx hf@latest download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
    --local-dir examples/data/LIBERO_LeRobot_v3

# LIBERO-10: point both variables at the suite directory.
export DATASET_PATH="$PWD/examples/data/LIBERO_LeRobot_v3/libero_10"
export LIBERO_ROOT="$DATASET_PATH"
bash examples/launch_sft_action_policy_libero_10_nano_p800.sh

# LIBERO-all: point both variables at the parent containing all four suites.
export DATASET_PATH="$PWD/examples/data/LIBERO_LeRobot_v3"
export LIBERO_ROOT="$DATASET_PATH"
bash examples/launch_sft_action_policy_libero_all_nano_p800.sh
```

Choose the intended recipe rather than running both sequentially by default.
The wrappers override the reference HSDP topology to shard 8 / replicate 1.
The [LIBERO training and closed-loop guide](../../../../../../../docs/action_policy_libero_posttrain.md)
describes the action normalization, camera layout and simulator/client contract.

## Paths and overrides

Paths are anchored to the framework repository root by the common example
launcher. Prefer absolute dataset/tokenizer paths, since `DROID_ROOT` and
`LIBERO_ROOT` are exported by the recipe wrappers before common path resolution.

| Variable | Purpose |
| --- | --- |
| `DATASET_PATH` | DROID parent, LIBERO suite or LIBERO-all parent |
| `DROID_ROOT` / `LIBERO_ROOT` | model dataset root; keep consistent with `DATASET_PATH` |
| `BASE_CHECKPOINT_PATH` | converted Cosmos3-Nano DCP |
| `WAN_VAE_PATH` | Wan2.2 VAE file |
| `TOKENIZER_PATH` | local Qwen3-VL-8B tokenizer metadata |
| `OUTPUT_ROOT` | log/run parent; defaults to each wrapper's timestamped path under `outputs/` |
| `IMAGINAIRE_OUTPUT_ROOT` | explicit artifact root; defaults to `OUTPUT_ROOT` |
| `CUDA_VISIBLE_DEVICES` | visible devices; eight P800s by default |
| `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` | torchrun topology |
| `ACTIVATION_CHECKPOINTING_MODE`, `ACTIVATION_CHECKPOINTING_SAVE_OPS_REGEX` | policy-wrapper activation checkpoint settings |

The existing P800 wrappers construct their own `TAIL_OVERRIDES` arrays.
`EXTRA_TAIL_OVERRIDES` is not consumed by the current common launcher, even
where older script comments mention it. For custom iteration limits or other
Hydra overrides, use the [raw training entry point](../../../../../../../docs/training.md#option-b-raw-torchrun)
and pass overrides after `--`, retaining the P800 settings from the selected
example. Changing process count alone does not adjust every parallelism axis.

## Outputs and export

Runs use `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/` and logs are under
`$OUTPUT_ROOT/logs/`. Keep the output root and resolved job identity stable for
a same-job restart; the wrappers' default timestamped roots create a new run.
The resolved `config.yaml` and `checkpoints/latest_checkpoint.txt` identify the
configuration and latest DCP.

```shell
# Set RUN_DIR to the actual job directory; this example assumes OUTPUT_ROOT below.
RUN_DIR=outputs/action_policy_libero_10/cosmos3/action_sft/action_policy_libero_10_nano
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

Use the actual job name from the chosen TOML/resolved config, rather than
assuming all action recipes share an output layout.
See the [training reference](../../../../../../../docs/training.md) and
[upstream action cookbook](../../../../../cookbooks/cosmos3/generator/action/finetune/README.md).
