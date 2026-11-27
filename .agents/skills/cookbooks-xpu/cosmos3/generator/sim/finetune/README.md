<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano Sim-Bimanual Fine-Tuning & Distillation — Kunlun P800

Use the P800 multi-stage launcher under `examples/` for the 4-stage Cosmos3-Nano
Sim-Bimanual training pipeline (bidirectional SFT → teacher forcing → TF-dCM
distillation → self-forcing DMD). This cookbook documents input preparation and
invocation; scripts and TOMLs are maintained only in `examples/`.
Run every command below **from the cosmos-framework repository root** with the
[XPU Sim-Bimanual environment](../../../README.md#cosmos-framework-kunlun-branch)
active (Action dependencies plus the Sim-Bimanual extras).

| Stage | Recipe               | P800 invocation                                                                                      | Paired TOML                                                                                                    |
| ----- | -------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `0`   | Bidirectional SFT    | [`launch_sft_sim_bimanual_p800.sh 0`](../../../../../../../examples/launch_sft_sim_bimanual_p800.sh) | [sim_bimanual_stage0_bidir.toml](../../../../../../../examples/toml/sft_config/sim_bimanual_stage0_bidir.toml) |
| `1`   | Teacher Forcing (TF) | [`launch_sft_sim_bimanual_p800.sh 1`](../../../../../../../examples/launch_sft_sim_bimanual_p800.sh) | [sim_bimanual_stage1_tf.toml](../../../../../../../examples/toml/sft_config/sim_bimanual_stage1_tf.toml)       |
| `2`   | TF-dCM Distillation  | [`launch_sft_sim_bimanual_p800.sh 2`](../../../../../../../examples/launch_sft_sim_bimanual_p800.sh) | [sim_bimanual_stage2_tfdcm.toml](../../../../../../../examples/toml/sft_config/sim_bimanual_stage2_tfdcm.toml) |
| `3`   | Self-Forcing DMD     | [`launch_sft_sim_bimanual_p800.sh 3`](../../../../../../../examples/launch_sft_sim_bimanual_p800.sh) | [sim_bimanual_stage3_sfdmd.toml](../../../../../../../examples/toml/sft_config/sim_bimanual_stage3_sfdmd.toml) |

The launcher enables offline HF access during training and does not download the
bundle, VAE, or checkpoints automatically. Prepare all inputs before launching.

## Prepare shared inputs and dataset bundle

Accept the required Hugging Face terms and authenticate with
`uvx hf@latest auth login` or `HF_TOKEN` first.

1. **Base Cosmos3-Nano DCP and Wan2.2 VAE:**

   ```shell
   export BASE_CHECKPOINT_PATH="$PWD/examples/checkpoints/Cosmos3-Nano-DCP"
   export WAN_VAE_PATH="$PWD/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"

   uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
       --local-dir "$(dirname "$WAN_VAE_PATH")"
   python -m cosmos_framework.scripts.convert_model_to_dcp \
       --checkpoint-path Cosmos3-Nano -o "$BASE_CHECKPOINT_PATH"
   ```

2. **Sim-Bimanual example training bundle:**

   ```shell
   mkdir -p examples/data/sim_bimanual
   HF_HUB_OFFLINE=0 uvx hf@latest download nvidia/Cosmos3-Nano-Sim-Bimanual \
       examples/training/sim_bimanual_robot_example_v1.tar.zst \
       --local-dir examples/data/sim_bimanual/downloads
   tar --zstd -xf \
       examples/data/sim_bimanual/downloads/examples/training/sim_bimanual_robot_example_v1.tar.zst \
       -C examples/data/sim_bimanual

   export DATASET_PATH="$PWD/examples/data/sim_bimanual/sim_bimanual_robot_example_v1"
   ```

Keep the P800 container's matching Torch/XPU stack and `LD_LIBRARY_PATH`.
For preparing custom bimanual datasets, see
[bimanual data preflight](../../../../../../../docs/bimanual_data_preflight.md)
and [bimanual session format](../../../../../../../docs/bimanual_sessions.md).

## Launching Stages 0–3

Pass the stage index (`0`, `1`, `2`, or `3`) as the first positional argument.
Unlike the Vision and Action wrappers, `launch_sft_sim_bimanual_p800.sh`
forwards any additional `key=value` CLI arguments directly into `TAIL_OVERRIDES`.

```shell
# Stage 0 — Bidirectional SFT (warm-starts from BASE_CHECKPOINT_PATH)
bash examples/launch_sft_sim_bimanual_p800.sh 0

# Stage 1 — Teacher Forcing (initializes from Stage 0 DCP)
export COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT="/path/to/stage0/iter_XXXXXXXXX"
bash examples/launch_sft_sim_bimanual_p800.sh 1

# Stage 2 — TF-dCM Distillation (student + frozen teacher from Stage 1 DCP)
export COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT="/path/to/stage1/iter_XXXXXXXXX"
bash examples/launch_sft_sim_bimanual_p800.sh 2

# Stage 3 — Self-Forcing DMD (student from Stage 2 DCP; teacher/critic from Stage 0 DCP)
export COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT="/path/to/stage2/iter_XXXXXXXXX"
bash examples/launch_sft_sim_bimanual_p800.sh 3
```

### P800 runtime defaults and practical notes

The launcher sets FlashAttention-2 (`COSMOS_ATTN_BACKENDS=flash2`), P800 fused
kernels, P800 VAE/memory switches, and `COSMOS_NATTEN_DENSE_FALLBACK=1` (see
[P800 runtime switches](../../../README.md#p800-runtime-switches)).
`COSMOS_FLASH2_LSE_BHFIX` and `COSMOS_FLASH2_VARLEN_PACKED_LSE` default to `0`
because the container's built-in `torch_xmlir/libxpu_flash_attention.so` already
produces standard LSE layouts — do **not** enable them with `torch_xmlir`'s `.so`,
as doing so misaligns LSE whenever `return_lse=True` is merged across batches or
chunks (e.g. teacher forcing / `merge_attentions`). Only override them to `1` if
an external/legacy `flash_attn` wheel providing `libxpu_flash_attention_train.so`
is installed in the environment.

Default `TAIL_OVERRIDES` in `launch_sft_sim_bimanual_p800.sh` are tuned for a
short smoke/validation run:

- `model.config.tokenizer.dtype=float16`, `trainer.callbacks.compile_tokenizer.enabled=false`
- `trainer.seed=42`, `dataloader_train.max_samples_per_batch=1`
- `trainer.logging_iter=1`, `trainer.grad_accum_iter=1`, `trainer.max_iter=12`
- `model.config.ema.enabled=false`

Append Hydra overrides directly on the command line for longer runs or stage handoff:

- **Saving checkpoints on P800 for cross-stage handoff:** disable async DCP
  (`checkpoint.dcp_async_mode_enabled=false`) to avoid pinned-memory allocation
  errors (`cudaError: 2`), and set `checkpoint.save_iter` and `trainer.max_iter`
  explicitly (note that the wrapper sets `COSMOS_SKIP_FINAL_CHECKPOINT=1`).
  If downstream stages expect EMA parameter trees, also re-enable EMA
  (`model.config.ema.enabled=true`):

  ```shell
  bash examples/launch_sft_sim_bimanual_p800.sh 0 \
      trainer.max_iter=100 \
      checkpoint.save_iter=100 \
      checkpoint.dcp_async_mode_enabled=false \
      model.config.ema.enabled=true
  ```

- **Deterministic mode:** `launch_sft_sim_bimanual_p800.sh` currently exports
  `COSMOS_DETERMINISTIC=0`. For deterministic precision-alignment runs, set
  `COSMOS_DETERMINISTIC=1` inside the launcher (or invoke the raw `torchrun`
  entry point with `--deterministic`).

## Paths and overrides

| Variable                                                              | Purpose                                                                      |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `DATASET_PATH`                                                        | local Sim-Bimanual bundle (exported as `COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE`)   |
| `BASE_CHECKPOINT_PATH`                                                | converted Cosmos3-Nano DCP (exported as `COSMOS_SIM_BIMANUAL_GA_CHECKPOINT`) |
| `WAN_VAE_PATH`                                                        | Wan2.2 VAE file                                                              |
| `COSMOS_SIM_BIMANUAL_STAGE0_CHECKPOINT`                               | Stage 0 DCP checkpoint (consumed by Stage 1 and Stage 3)                     |
| `COSMOS_SIM_BIMANUAL_STAGE1_CHECKPOINT`                               | Stage 1 DCP checkpoint (consumed by Stage 2)                                 |
| `COSMOS_SIM_BIMANUAL_STAGE2_CHECKPOINT`                               | Stage 2 DCP checkpoint (consumed by Stage 3)                                 |
| `OUTPUT_ROOT`                                                         | log/run parent; defaults to `outputs/sim_bimanual/<stage>_<timestamp>`       |
| `IMAGINAIRE_OUTPUT_ROOT`                                              | explicit artifact root; defaults to `OUTPUT_ROOT`                            |
| `CUDA_VISIBLE_DEVICES`                                                | visible devices; eight P800s by default                                      |
| `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` | torchrun topology                                                            |

## Outputs and checkpoint export

Runs write to `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/` and console
logs are under `$OUTPUT_ROOT/logs/`. Set `OUTPUT_ROOT` explicitly for a stable
job directory:

```shell
OUTPUT_ROOT="$PWD/outputs/sim_bimanual_stage0_p800" \
    bash examples/launch_sft_sim_bimanual_p800.sh 0 \
    trainer.max_iter=100 checkpoint.save_iter=100 \
    checkpoint.dcp_async_mode_enabled=false
```

To export a trained DCP checkpoint to safetensors:

```shell
RUN_DIR=outputs/sim_bimanual_stage0_p800/cosmos3/sim_bimanual/sim_local_bimanual_stage0_bidir
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

Use the actual job name from the selected stage TOML (`sim_local_bimanual_stage0_bidir`,
`sim_local_bimanual_stage1_tf`, `sim_local_bimanual_stage2_tfdcm`, or
`sim_local_bimanual_stage3_sfdmd`). See
[bimanual inference](../../../../../../../docs/inference_bimanual.md) and the
[training reference](../../../../../../../docs/training.md).
