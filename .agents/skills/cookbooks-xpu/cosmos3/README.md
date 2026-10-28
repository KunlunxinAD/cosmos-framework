<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 XPU Cookbooks: Environment Setup

Shared environment setup for every Cosmos3 cookbook adapted to **Kunlun P800 (XPU)**. These are the
P800 counterparts of the [NVIDIA Cosmos3 cookbooks](../../cookbooks/cosmos3/README.md): the same
`cosmos_framework.scripts.train` flow and the recipe TOMLs maintained in `examples/`.
The existing P800 examples target a single eight-device node with the fused kernels enabled. Each recipe README links back here for
setup and for the P800 switch reference.

| Backend                                                             | Use it for                                                | Used by                         |
| ------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------- |
| [Cosmos Framework (Kunlun branch)](#cosmos-framework-kunlun-branch) | Native PyTorch training on P800, launched with `torchrun` | Generator (Audiovisual, Action) |

The cookbook contains usage documentation. All executable launchers and TOMLs
are maintained under `examples/`; run them from the framework repository root
after preparing the required local inputs. Super's dedicated P800 launcher is pending.

| Cookbook | Recipes | Where |
| --- | --- | --- |
| Audiovisual SFT | Nano, Edge, Nano map-style; Super LoRA P800 entry pending | [Vision usage](generator/audiovisual/finetune/README.md) |
| Action SFT | DROID policy, DROID forward dynamics, LIBERO-10, LIBERO-all | [Action usage](generator/action/finetune/README.md) |

## Prerequisites

- Linux host with **Kunlun P800 (XPU)** access (a single 8-GPU node for these recipes).
- **P800 container image.** These recipes are validated inside the Kunlun XAV image, which ships the
  matching P800 `torch` + `torch_xmlir` stack. Pull it with:

  ```shell
  docker pull hub.kunlunxin.com/public/kunlunxin-self-driving/xav:26.08-py312-torch29
  ```

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `git`, and `git-lfs` installed.
- Hugging Face access to the gated Cosmos3 model repos and datasets. Every model and dataset is
  pulled from its **official Hugging Face repo** (`nvidia/*`, `Wan-AI/*`, `Qwen/*`) — no mirror is
  required. Authenticate once before the first run:

  ```shell
  uvx hf@latest auth login
  # or:
  export HF_TOKEN=<your_token>
  ```

- Enough local disk for the venv, the base checkpoint, and the datasets (the DROID splits are
  TB-scale; the small inputs — VAE + tokenizer + DCP — are tens of GiB).

## Cosmos Framework (Kunlun branch)

These recipes use the P800 fused kernels in the
[Kunlun framework branch](https://github.com/KunlunxinAD/cosmos-framework/tree/kunlun_dev).
See the
[Cosmos XPU Cookbook](https://github.com/KunlunxinAD/cosmos/blob/kunlun_dev/COOKBOOK%E2%80%94XPU.md)
for the recipe index. Prepare the environment below from the framework repository root.
XPU usage documentation lives in `.agents/skills/cookbooks-xpu/cosmos3/`.
The launchers and paired TOMLs live in `examples/` alongside the framework training examples.

1. **Keep the host P800 Torch/XPU stack.** The P800 container image above already ships a matching
   `torch` + `torch_xmlir` build. Install the framework's Python dependencies **without** the CUDA
   dependency groups — do **not** pull a `cu128`/`cu130` wheel, it will replace the working XPU Torch.
2. **Install the extra dependencies.** The base image does not ship the framework's Python
   requirements. Install `ffmpeg` and the Python packages for the recipe family you are running —
   the audiovisual and action sets pin a few packages differently, so use a **separate venv per
   family**. Training runs from the checkout via `PYTHONPATH` (the launch scripts set it for you).

   System package (both families):

   ```shell
   apt-get update && apt-get install -y ffmpeg
   ```

   **Audiovisual** (Nano / Edge / Super LoRA / Nano map-style) — from the framework repository root,
   install the package (without the CUDA groups) and then the audiovisual extras:

   ```shell
   python -m pip install . --upgrade-strategy only-if-needed
   python -m pip install \
     loguru==0.7.3 iopath==0.1.10 diffusers==0.39.0 hydra-core==1.3.5 omegaconf==2.3.1 \
     transformers==4.57.6 peft==0.20.0 boto3==1.43.75 botocore==1.43.75 cattrs==26.1.0 \
     ImageIO==2.37.4 imageio-ffmpeg==0.6.0 msgpack==1.2.1 obstore==0.11.0 \
     nvidia-cudnn-frontend==1.27.0
   python -m pip install --no-deps --force-reinstall wandb==0.27.2
   python -m pip install 'GitPython>=3.1.29'
   ```

   **Action** (DROID / LIBERO) — from the framework repository root, install the
   package (without the CUDA groups) and then the action extras:

   ```shell
   python -m pip install . --upgrade-strategy only-if-needed
   python -m pip install \
     'transformers>=4.57.1,<5.0.0' 'peft>=0.17.0,<1.0' 'datasets==4.6.1' \
     'boto3==1.40.61' 'botocore==1.40.61' 'aiobotocore==2.25.1' 'aioboto3==15.5.0' \
     's3fs==2026.2.0' 'fsspec==2026.2.0' 'iopath>=0.1.10' 'tensorstore==0.1.81' \
     'wandb==0.27.2' 'nvdlfw-inspect' 'onnxscript' 'blobfile' 'deepdiff>=7.0.1,<9.0.0' \
     'einx' 'ftfy' 'fvcore' 'kornia' 'lz4' 'megatron-core' 'more-itertools' 'ninja' \
     'qwen-vl-utils' 'tiktoken' 'timm' 'typeguard' 'webdataset'
   # lerobot 0.4.4's metadata would downgrade the diffusers/torch stack; install it without deps.
   python -m pip install --no-deps 'lerobot==0.4.4'
   ```

Preserve the container's `LD_LIBRARY_PATH`; remove an inherited entry only after
identifying a conflicting library. For CPU placement, see the
[Kunlun NUMA affinity guide](../../../../docs/numa_cpu_affinity.md).

Prepare the datasets, VAE, matching processor/tokenizer and base DCP using the
vision/action guides before launching. The example helper resolves repository-relative
paths from `examples/`; the P800 wrappers enable offline access during training.

## P800 runtime switches

The framework fused-kernel switches default to the reference implementation.
The P800 example scripts explicitly set the following switches to `1`; a parent-shell
assignment of `0` is overwritten by those scripts. For an A/B run, use a direct
training invocation with the intended settings. Preserve the container library paths.

### Fused kernels (all recipes)

| Switch                         | Default | Effect                          |
| ------------------------------ | ------- | ------------------------------- |
| `XDNN_USE_FAST_SWISH`          | `1`     | XDNN fast SwiSH/SiLU activation |
| `COSMOS_FUSED_RMSNORM`         | `1`     | fused RMSNorm                   |
| `COSMOS_FUSED_ROPE`            | `1`     | fused rotary position embedding |
| `COSMOS_FUSED_CAUSAL_CONV_PAD` | `1`     | fused causal-conv padding       |
| `COSMOS_FUSED_VAE_NORM`        | `1`     | fused VAE normalization         |

### Runtime / determinism (all recipes)

| Switch                                    | Default                    | Effect                                               |
| ----------------------------------------- | -------------------------- | ---------------------------------------------------- |
| `COSMOS_TRAINING`                         | `1`                        | select the training runtime path                     |
| `CUDA_VISIBLE_DEVICES`                    | `0,1,2,3,4,5,6,7`          | the 8 P800 devices to use                            |
| `PYTHONHASHSEED`                          | `42`                       | reproducible hashing (paired with `trainer.seed=42`) |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1`                        | require local inputs before training                 |


The P800 entry points are `examples/launch_sft_*_p800.sh`; their paired TOMLs
are under `examples/toml/sft_config/`. Their runtime overrides take precedence over
the TOMLs, and their current common helper does not read `EXTRA_TAIL_OVERRIDES`.
See [vision usage](generator/audiovisual/finetune/README.md),
[action usage](generator/action/finetune/README.md), the
[SFT examples](../../../../examples/README.md) and the
[training reference](../../../../docs/training.md). Super's P800 launcher will be
added separately; no replacement script is created by this documentation cleanup.
