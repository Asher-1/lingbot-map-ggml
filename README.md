<div align="center">
  <img src="assets/teaser.webp" width="100%">

<h1>LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction</h1>

Robbyant Team

</div>

<div align="center">

[![Conference Version Paper](https://img.shields.io/static/v1?label=Conference%20Version&message=Paper&color=blue)](https://linzhuo.xyz/gct.pdf)
[![Technical Report Version Paper](https://img.shields.io/static/v1?label=Technical%20Report%20Version&message=Paper&color=red)](lingbot-map_paper.pdf)
[![Project](https://img.shields.io/badge/Project-Website-blue)](https://technology.robbyant.com/lingbot-map)
[![HuggingFace](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange)](https://huggingface.co/robbyant/lingbot-map)
[![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple)](https://www.modelscope.cn/models/Robbyant/lingbot-map)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE.txt)

</div>

https://github.com/user-attachments/assets/fe39e095-af2c-4ec9-b68d-a8ba97e505ab

-----

### 🗺️ Meet LingBot-Map! We've built a feed-forward 3D foundation model for streaming 3D reconstruction! 🏗️🌍

LingBot-Map has focused on:

- **Geometric Context Transformer**: Architecturally unifies coordinate grounding, dense geometric cues, and long-range drift correction within a single streaming framework through anchor context, pose-reference window, and trajectory memory.
- **High-Efficiency Streaming Inference**: A feed-forward architecture with paged KV cache attention, enabling stable inference at ~20 FPS on 518×378 resolution over long sequences exceeding 10,000 frames.
- **State-of-the-Art Reconstruction**: Superior performance on diverse benchmarks compared to both existing streaming and iterative optimization-based approaches.

---

## 📑 Table of Contents

<details>
<summary>Click to expand</summary>

- [📑 Table of Contents](#-table-of-contents)
- [📰 News](#-news)
- [📋 TODO](#-todo)
- [⚙️ Installation](#️-installation)
- [📦 Model Download](#-model-download)
- [🚀 Quick Start](#-quick-start)
  - [From Clone to End-to-End](#from-clone-to-end-to-end)
  - [C++ GGML runtime](#c-ggml-runtime)
    - [GGML deployment: CUDA and Vulkan](#ggml-deployment-cuda-and-vulkan)
    - [GGML GUI reconstruction (one command)](#ggml-gui-reconstruction-one-command)
- [🎬 Interactive Demo (`demo.py`)](#-interactive-demo-demopy)
  - [Try the Example Scenes](#try-the-example-scenes)
    - [🎯 Featured: indoor walkthrough (~25 000 frames, 13 minutes)](#-featured-indoor-walkthrough-25-000-frames-13-minutes)
  - [Dynamic Demo (From Droid-W)](#dynamic-demo-from-droid-w)
  - [Streaming with Keyframe Interval](#streaming-with-keyframe-interval)
  - [Windowed Inference (for long sequences, \>3000 frames)](#windowed-inference-for-long-sequences-3000-frames)
  - [Sky Masking](#sky-masking)
  - [Visualization Options](#visualization-options)
  - [Performance \& Memory](#performance--memory)
    - [Without FlashInfer (SDPA fallback)](#without-flashinfer-sdpa-fallback)
    - [Running on Limited GPU Memory](#running-on-limited-gpu-memory)
    - [Faster Inference](#faster-inference)
- [🎥 Offline Rendering Pipeline (`demo_render/batch_demo.py`)](#-offline-rendering-pipeline-demo_renderbatch_demopy)
  - [Install (extends the main install)](#install-extends-the-main-install)
  - [Worked Example — long indoor walkthrough (~25 000 frames, 13 minutes)](#worked-example--long-indoor-walkthrough-25-000-frames-13-minutes)
    - [Quick Mode and Demo Reproduction](#quick-mode-and-demo-reproduction)
  - [Worked Example — outdoor drive scene](#worked-example--outdoor-drive-scene)
  - [Worked Example — LingBot-World scenes](#worked-example--lingbot-world-scenes)
  - [Camera Path (YAML)](#camera-path-yaml)
    - [YAML structure](#yaml-structure)
    - [Available modes](#available-modes)
    - [Single-shot YAML examples](#single-shot-yaml-examples)
  - [Output files](#output-files)
- [📜 License](#-license)
- [📖 Citation](#-citation)
- [✨ Acknowledgments](#-acknowledgments)

</details>

---

## 📰 News

- **2026-06-28** — Fixed an SDPA KV cache bug. **The SDPA backend now performs better on long sequences**. We still recommend the FlashInfer backend for the best performance.
- **2026-05-25** — 📊 **Evaluation benchmark released**. We released the evaluation scripts for KITTI and Oxford Spires — see [benchmark/](benchmark/) for the pipeline, and run [`preprocess/oxford.py`](preprocess/oxford.py) to prepare Oxford Spires data before evaluation.
- **2026-04-29** — 📹 **Long-video demo released**. We released a very-long-video example (~25 000 frames, 13-minute indoor walkthrough) rendered with the offline pipeline — see [Worked Example](#worked-example--long-indoor-walkthrough-25-000-frames-13-minutes) for the command, flag rationale, and rendered output.
- **2026-04-27** — 🚀 **LingBot-Map accelerated**. Pull the latest `main` and run `python demo.py --compile ...` or `python gct_profile.py --backend flashinfer --dtype bf16 --compile` to verify on your hardware.
- **2026-04-24** — Fixed a FlashInfer KV cache bug where `--keyframe_interval > 1` silently cached non-keyframes. **You should now see better pose and reconstruction quality when running with more than 320 frames**.

---

## 📋 TODO

- ✅ Release evaluation benchmark
  - ✅ Oxford Spires dataset
  - ✅ KITTI dataset
  - ✅ VBR dataset
  - ✅ Droid-W dataset
  - ✅ TUM-D dataset
  - ✅ 7-scenes dataset
  - ✅ ETH3D dataset
  - ✅ Tanks and Temples dataset
  - ✅ NRGBD dataset
- ✅ Release demo scripts
  - ✅ Indoor long-video demo ([Featured indoor walkthrough](#-featured-indoor-walkthrough-25-000-frames-13-minutes))
  - ✅ Outdoor long-video demo
  - ✅ LingBot-World demo ([Worked example](#worked-example--lingbot-world-scenes))
  - ✅ Aerial long-video demo

---

## ⚙️ Installation

**1. Create conda environment**

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
```

**2. Install PyTorch (CUDA 12.8)**

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

> PyTorch 2.8.0 is the recommended version because NVIDIA Kaolin (required by the batch rendering pipeline) has prebuilt wheels for `torch-2.8.0_cu128`. If you only need `demo.py` you may use a newer PyTorch, but the batch renderer then requires building Kaolin from source.
> For other CUDA versions, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

**3. Install lingbot-map**

```bash
pip install -e .
```

**4. Install FlashInfer (recommended)**

FlashInfer provides paged KV cache attention for efficient streaming inference. It is a pure-Python package that JIT-compiles CUDA kernels on first use, so a single wheel works across CUDA/PyTorch versions:

```bash
pip install --index-url https://pypi.org/simple flashinfer-python
```

> `--index-url https://pypi.org/simple` is only needed if your default pip index is an internal mirror that doesn't have `flashinfer-python`.
> (Optional) For faster first-use, you can additionally install a CUDA-specific JIT cache: `pip install flashinfer-jit-cache -f https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/`.
> See [FlashInfer installation](https://docs.flashinfer.ai/installation.html) for details. If FlashInfer is not installed, the model falls back to SDPA (PyTorch native attention) via `--use_sdpa`.

**5. Visualization dependencies (optional)**

```bash
pip install -e ".[vis]"
```

## 📦 Model Download

| Model Name | Huggingface Repository | ModelScope Repository | Description |
| :--- | :--- | :--- | :--- |
| lingbot-map | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | Balanced checkpoint (used in paper, benchmark and offline demo) — trade off all-around performance across short and long sequences. |
| lingbot-map-long | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | Long-sequence checkpoint — same architecture as `lingbot-map`, converted to `lingbot-map-long-{f32,f16,q8}.gguf` for the GGML engine (see `cpp_ggml/models/MODEL_CARD.md`). |
| lingbot-map-stage1 | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | Stage-1 training checkpoint of lingbot-map — can be loaded into the VGGT model for bidirectional inference (c2w). |

> 🚧 **Coming soon:** we're training an stronger model that supports longer sequences — stay tuned.

## 🚀 Quick Start

### From Clone to End-to-End

The complete path from a fresh machine to a running reconstruction and a
validated end-to-end gate:

```bash
# 1. Clone with the pinned ggml submodule (required by the C++ runtime)
git clone --recursive https://github.com/<you>/lingbot-map-ggml.git
cd lingbot-map-ggml
git submodule update --init --recursive   # if cloned without --recursive

# 2a. Python environment — official PyTorch engine (also runs the GGML GUI)
conda create -n lingbot-map python=3.10 -y && conda activate lingbot-map
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[vis]" && pip install --index-url https://pypi.org/simple flashinfer-python

# 2b. ...or a minimal viewer-only env for the GGML engine (no torch needed)
#     any python with:  pip install viser trimesh numpy pillow

# 3. Weights — download at least one of:
huggingface-cli download robbyant/lingbot-map \
  lingbot-map.pt --local-dir cpp_ggml/models/pytorch          # official checkpoint (PyTorch engine)
huggingface-cli download robbyant/lingbot-map \
  lingbot-map-long.pt --local-dir cpp_ggml/models/pytorch     # long-sequence checkpoint (optional)
huggingface-cli download Asher-1/lingbot-map-gguf \
  lingbot-map-q8.gguf lingbot-map-f16.gguf --local-dir cpp_ggml/models/gguf  # GGML engine
# (f32/f16 GGUFs can also be converted locally from the .pt:
#  python cpp_ggml/scripts/convert_lingbot.py cpp_ggml/models/pytorch/lingbot-map.pt \
#    cpp_ggml/models/gguf/lingbot-map-f32.gguf --outtype f32
#  long-checkpoint GGUFs ship as lingbot-map-long-{f32,f16,q8}.gguf in the
#  same GGUF repo; select them with --gguf or GGML_MODEL)

# 4. One-command GUI (auto-picks the engine per what this machine has)
bash run_gui.sh
#    -> http://localhost:8080
#    --engine ggml|pytorch forces one; --frames N for a quick look

# 5. Headless end-to-end validation (builds + runs the 286-frame parity gate)
bash cpp_ggml/scripts/run_e2e.sh cuda q8      # or: vulkan f16
#    -> PASS lines + CSV/PNG/PLY evidence in cpp_ggml/benchmarks/
```

Notes:
- The GGML engine needs **no torch**: the GUI only requires `viser`, `numpy`
  and `pillow` in whatever interpreter it finds (project venv, system python,
  any conda env; `viser trimesh` are auto-installed when missing).
- GPU memory at the default `scale=8/window=64` profile: q8 ≈ 21 GiB,
  f16 ≈ 22.4 GiB at 518×294; 12 GiB cards should use the bounded
  `scale=1/window=4` profile (`--kv_cache_scale 1 --kv_cache_window 4`) or
  the run_e2e f16/Vulkan path.
- Accuracy vs the official PyTorch pipeline: f16 GGUF is the full-alignment
  format (pose 1.72e-04 / depth 4.72e-04 over 286 frames); q8 trades part of
  that margin for half the memory. Details in
  [`cpp_ggml/benchmarks/validation_report.md`](cpp_ggml/benchmarks/validation_report.md).

### C++ GGML runtime

The repository includes a pinned GGML v0.21.0 integration under
[`cpp_ggml/`](cpp_ggml/README.md). It provides CMake patch replay, checkpoint
conversion to GGUF, CPU/CUDA/Vulkan backend selection, and measurement scripts.

#### GGML deployment: CUDA and Vulkan

Download q8 or f16 GGUF deployment models from
[Asher-1/lingbot-map-gguf](https://huggingface.co/Asher-1/lingbot-map-gguf/tree/main)
into `cpp_ggml/models/gguf/`. The C++ graph runs DINO/GCT streaming attention,
RoPE, persistent global/camera caches, CameraCausalHead, and four-scale DPT
depth reconstruction without cuDNN. On the official 286-frame `courthouse`
sequence at the native aspect (518×294, the identity output of the official
crop rule for these frames) with the upstream `scale=8/window=64` profile and
the persistent F16 KV cache, q8 CUDA has strict GGUF-reference parity
(pose RMSE 6.14e-05, depth RMSE 9.75e-05; zero pose violations, ≤0.010%
depth violations at discontinuities). Vulkan q8 reaches 3.89e-05 / 9.71e-05
at the same profile. These runs use the full C++ postprocess path, no cuDNN,
and the same cache profile as the official PyTorch pipeline. Run the
reproducible command below rather than reusing cached plots:

```bash
bash cpp_ggml/scripts/run_e2e.sh cuda q8
```

**End-to-end alignment with the official PyTorch pipeline**: preprocessing
is bit-identical (`max_abs_diff=0`), and against the official fp32/bf16
checkpoint the f16 GGUF deviates by pose `1.72e-04` / depth `4.72e-04` over
the full 286-frame stream (the GUI default model; older checkouts can force it
with `--gguf cpp_ggml/models/gguf/lingbot-map-f16.gguf`) —
two orders of magnitude below the official bf16 deployment self-noise, i.e.
the GGML engine matches the Python reconstruction. The q8 default carries
the q8 quantization loss instead (`1.31e-03` pose / `3.16e-03` depth over
286 frames), which is the same loss PyTorch itself incurs decoding the same
q8 weights — the engine is fully aligned; only the weight format differs.
Full alignment matrix: `cpp_ggml/benchmarks/validation_report.md`.

Vulkan q8 and f16 both pass the complete 286-frame gate. Run
`bash cpp_ggml/scripts/run_e2e.sh vulkan f16` to reproduce the f16 result.

**Fast mode (`--kv-f16 flash`)**: the same streams run through tensor-core
flash attention with F32-effective PV numerics — F32 VKQ accumulation + P
hi/lo split on CUDA, coopmat1 FA + P hi/lo + F32 output chain on Vulkan with
scalar GEMMs — plus an F32 camera-trunk cache. It holds the same 1e-4-class
parity at FlashInfer-class speed: over the full 286-frame stream, CUDA f16
lands at pose 1.39e-04 / depth 4.55e-04 in **2m11s** (the PyTorch FlashInfer
reference needs 2m14s) and Vulkan f16 at pose 7.27e-05 / depth 1.06e-04 in
**2m39s**; f32 is the deepest reference (7.25e-05 / 1.35e-04 on Vulkan) and
q8 carries only the documented q8 weight cost. The default strict mode
remains the bit-level validation route (6m47s CUDA / 7m05s Vulkan).
Engineering detail: `AGENTS.md`; measured matrix:
`cpp_ggml/benchmarks/validation_report.md`.

The detailed setup, limitations, model card, and generated evidence live in
[`cpp_ggml/README.md`](cpp_ggml/README.md) and
[`cpp_ggml/models/MODEL_CARD.md`](cpp_ggml/models/MODEL_CARD.md).

#### GGML GUI reconstruction (one command)

The same official viser reconstruction GUI as `demo.py` — RGB-D point clouds,
camera frustums, trajectory, playback — driven by either the native GGML
runtime or the official PyTorch pipeline, with one command and the same
preprocessing / streaming profile on both sides, so the two reconstructions
are directly comparable:

```bash
bash run_gui.sh                                        # auto engine, courthouse, all frames
bash run_gui.sh --engine ggml --frames 40              # native C++ engine, quick look
bash run_gui.sh --engine pytorch --image_folder example/loop   # official demo.py pipeline
bash run_gui.sh --engine ggml --backend CUDA0          # picks build-cuda automatically
bash run_gui.sh --engine ggml --backend CUDA0 --build cpp_ggml/build-cuda

# Outdoor scenes with sky in view — mask_sky runs the NATIVE skyseg GGUF
# inside the C++ CLI (no onnxruntime); masks are cached as PNGs and
# original|mask|overlay panels are written next to the repo:
bash run_gui.sh --engine ggml --mask_sky   --sky_mask_dir example/courthouse_ggml_sky_masks   --sky_mask_visualization_dir example/courthouse_sky_vis
```

The default `--engine auto` picks the official PyTorch pipeline when this
machine has the `.pt` checkpoint and a torch environment, and the native
GGML runtime otherwise; force one with `--engine ggml|pytorch`. The script
auto-builds the C++ runtime on first run, probes the project venv, the
system interpreters and every conda install for the required dependencies
(`pip install viser trimesh` auto-installed when only those are missing),
translates the shared convenience flags between the engines
(`--frames` <-> demo.py's `--first_k`), auto-selects the SDPA attention
backend when flashinfer is not installed, and opens `http://localhost:8080`.
`--mask_sky` with the GGML engine runs a NATIVE ggml sky-segmentation model
(`lingbot-map-skyseg-f16.gguf`, 100% mask agreement with the official
onnxruntime path over the 8-frame courthouse stream, ~17.6 ms/frame on
CUDA0 with an RTX 4090) — no onnxruntime required. The GGML engine streams: a live viser view
grows point clouds and camera frustums frame-by-frame while the native
runtime runs, then the full official viewer takes over on the same port when
inference completes (disable the live phase with `--no-streaming_view`).
Both engines default to the official crop rule
(`--image_size 518`, aspect-preserving, snapped to the patch grid) and the
upstream `scale=8/window=64` cache profile with the persistent F16 KV cache.
Useful variants: `GGML_MODEL=cpp_ggml/models/gguf/lingbot-map-f16.gguf`
for the higher-fidelity format, `--backend CUDA0 --build cpp_ggml/build-cuda`
for CUDA. The `cpp_ggml/scripts/run_gui.sh` entry point is a thin forwarder
with `--engine ggml` pre-selected. See the model card for the full capacity
limits.

After installation, run your first scene with one command:

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```

This launches an interactive [viser](https://github.com/nerfstudio-project/viser) viewer at `http://localhost:8080`. See [Interactive Demo](#-interactive-demo-demopy) below for the full set of scenes and flags, or jump to [Offline Rendering Pipeline](#-offline-rendering-pipeline-demo_renderbatch_demopy) for long-sequence batch rendering.

## 🎬 Interactive Demo (`demo.py`)

Run `demo.py` for interactive 3D visualization via a browser-based [viser](https://github.com/nerfstudio-project/viser) viewer (default `http://localhost:8080`).

Images are preprocessed with canonical crop to `--image_size` (square 518).
The upstream `scale=8/window=64` profile runs end to end on a 24 GiB GPU when
[FlashInfer](https://docs.flashinfer.ai/) is installed (bf16 paged KV cache,
measured ~15-17 GB steady state); without it the SDPA fallback stores the K
cache in fp32 and sits at ~21 GB — installable via
`pip install flashinfer-python`, otherwise add `--use_sdpa`:

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/courthouse --use_sdpa
```

### Try the Example Scenes

We provide three example scenes in `example/` that you can run out of the box:
```bash
# courthouse scene
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```


https://github.com/user-attachments/assets/aa10f7ab-8024-43c7-92f8-d56159ec85c8






```bash
# University scene
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/university --mask_sky
```


https://github.com/user-attachments/assets/212a1744-6ff5-4ccf-9bd4-728608248b57







```bash
# Loop scene (loop closure trajectory)
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/loop
```


https://github.com/user-attachments/assets/5ae0a292-b081-40c6-838c-b7c1a0538d75





#### 🎯 Featured: indoor walkthrough (~25 000 frames, 13 minutes)


*Sequence is too long for the interactive viser viewer — this clip was rendered with the [Offline Rendering Pipeline](#-offline-rendering-pipeline-demo_renderbatch_demopy). See that section for the full command.*

We will provide more examples in the follow-up.

### Dynamic Demo (From Droid-W)

**Dataset:** Download the demo sequences from [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) on Hugging Face.

Example run on the `dynamic` sequence from the dataset above (sky masking on, 4 camera optimization iterations, keyframe every 2 frames):

Run the `dynamic` sequence with sky masking, 4 camera optimization iterations, and an input stride of 2:

```bash
python demo.py \
    --image_folder /path/to/dynamic\
    --model_path ../../Lingbot-Map/lingbot-map.pt \
    --camera_num_iterations 4 \
    --mask_sky \
    --stride 2
```



https://github.com/user-attachments/assets/567b6e9b-1cbf-402a-96be-9bab70715ec3

<img width="1453" height="1195" alt="image" src="https://github.com/user-attachments/assets/27f8c6b7-339e-4e5f-9776-7cb577147401" />





### Streaming with Keyframe Interval

Use `--keyframe_interval` to reduce KV cache memory by only keeping every N-th frame as a keyframe. Non-keyframe frames still produce predictions but are not stored in the cache. This is useful for long sequences which exceed 320 frames (We train with video RoPE on 320 views, so performance degrades when the KV cache stores more than 320 views. Using a keyframe strategy allows inference over longer sequences.). In demo.py, the keyframe interval is calculated automatically.

> **Note on inference range.** Our method does not perform state resetting by default, so the maximum inference range is bounded by the longest distance seen during training on the dataset. Beyond that distance, state resetting becomes necessary. If you observe pose collapse, switch to windowed mode (`--mode windowed`) — in most cases tuning `--keyframe_interval` alone is enough and the rest of the windowed parameters can stay at their defaults.


### Windowed Inference (for long sequences, >3000 frames)

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --video_path video.mp4 --fps 10 \
    --mode windowed --window_size 128 --overlap_keyframes 16 --keyframe_interval 2 
```


### Sky Masking

Sky masking uses an ONNX sky segmentation model to filter out sky points from the reconstructed point cloud, which improves visualization quality for outdoor scenes.

**Setup:**

```bash
# Install onnxruntime (required)
pip install onnxruntime        # CPU
# or
pip install onnxruntime-gpu    # GPU (faster for large image sets)
```

By default, root `demo.py` resolves the sky segmentation model as `skyseg.onnx` relative to the current working directory. If that default file is missing, it is automatically downloaded from [HuggingFace](https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx) on first use. If the download fails or does not produce a regular file, sky masking stops with a `RuntimeError` that reports the model path, download URL, cause, and manual setup guidance; it never silently continues without masking.

For manual recovery while keeping the default path, download `skyseg.onnx` into the directory from which you run `demo.py`:

```bash
wget -O skyseg.onnx https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky
```

**Usage with an explicit model path:**

To use a model stored elsewhere, pass its absolute path with `--sky_model`:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky \
    --sky_model /absolute/path/to/skyseg.onnx
```

Sky masks are cached in `<image_folder>_sky_masks/` so subsequent runs skip regeneration. You can also specify a custom cache directory with `--sky_mask_dir`, or save side-by-side mask visualizations with `--sky_mask_visualization_dir`:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky \
    --sky_mask_dir /path/to/cached_masks/ \
    --sky_mask_visualization_dir /path/to/mask_viz/
```

### Visualization Options

| Argument | Default | Description |
|:---|:---|:---|
| `--port` | `8080` | Viser viewer port |
| `--conf_threshold` | `1.5` | Visibility threshold for filtering low-confidence points |
| `--point_size` | `0.00001` | Point cloud point size |
| `--downsample_factor` | `10` | Spatial downsampling for point cloud display |

### Performance & Memory

#### Without FlashInfer (SDPA fallback)

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --use_sdpa
```

#### Running on Limited GPU Memory

If you run into out-of-memory issues, try one (or both) of the following:

- **`--offload_to_cpu`** — offload per-frame predictions to CPU during inference (on by default; use `--no-offload_to_cpu` only if you have memory to spare).
- **`--num_scale_frames 2`** — reduce the number of bidirectional scale frames from the default 8 down to 2, which shrinks the activation peak of the initial scale phase.

#### Faster Inference

Lower the number of iterative refinement steps in the camera head to trade a small amount of pose accuracy for wall-clock speed:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --camera_num_iterations 1
```

`--camera_num_iterations` defaults to `4`; setting it to `1` skips three refinement passes in the camera head (and shrinks its KV cache by 4×).

## 🎥 Offline Rendering Pipeline (`demo_render/batch_demo.py`)

Use this pipeline when your sequence is too long for the interactive viser viewer — for example, the [indoor walkthrough featured above](#-featured-indoor-walkthrough-25-000-frames-13-minutes). `demo_render/batch_demo.py` is the all-in-one offline entry point: feed it a video or a folder of images and it will run model inference and produce a headless point-cloud flythrough MP4 in a single command. It shares the same PyTorch / FlashInfer / checkpoint stack as `demo.py`.

For those constrained by limited VRAM or GPU usage, you may also refer to the implementation at: https://github.com/ureeey/lingbot-map-rtx4060-8g/commit/eeee84a89cc97c1e39b736b46df4ee315275700b

### Install (extends the main install)

**1. Rendering Python dependencies**

```bash
pip install -e ".[vis,render]"
```

`render` pulls in `open3d>=0.19` and `pyyaml` (the core `numpy<2`
constraint comes from the base `lingbot-map` install). Sky masking in this
pipeline uses `onnxruntime-gpu` with the dynamic-batch
`skyseg_batch.onnx` published in the
[`robbyant/lingbot-map`](https://huggingface.co/robbyant/lingbot-map)
model repository:

```bash
pip install onnxruntime-gpu
wget -O skyseg_batch.onnx \
  https://huggingface.co/robbyant/lingbot-map/resolve/main/skyseg_batch.onnx
```

The offline renderer downloads `skyseg_batch.onnx` automatically when its
configured path is missing. Use
`--skyseg_model_path /absolute/path/to/skyseg_batch.onnx` with
`demo_render/batch_demo.py`; for standalone
`demo_render/rgbd_scan_render.py`, use
`--sky_model /absolute/path/to/skyseg_batch.onnx` or set
`preprocess.sky_model` in YAML. The single-image root `demo.py` continues to
use `skyseg.onnx`.

**2. Kaolin** — matches the PyTorch 2.8.0 + CUDA 12.8 recommended above:

```bash
pip install --index-url https://pypi.org/simple \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
```

> `--index-url https://pypi.org/simple` bypasses any internal mirror that might otherwise serve the PyPI placeholder wheel (which raises `ImportError` on import).
> NVIDIA Kaolin does not publish prebuilt wheels for PyTorch 2.9.x — if you're on 2.9 for other reasons, build Kaolin from source (`pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git`, needs local CUDA toolkit). For other torch/CUDA combinations see [NVIDIA Kaolin installation](https://kaolin.readthedocs.io/en/latest/notes/installation.html).

**3. ffmpeg**

```bash
sudo apt install ffmpeg    # or: brew install ffmpeg
```

**4. CUDA extensions** (required before first run)

```bash
cd demo_render/render_cuda_ext && python setup.py build_ext --inplace && cd ../..
```

This builds `voxel_morton_ext` and `frustum_cull_ext` in place — both are imported by `rgbd_render` for GPU voxelization and frustum culling.

### Worked Example — long indoor walkthrough (~25 000 frames, 13 minutes)

**Dataset:** Download the example video from [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) on Hugging Face.

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/indoor_travel.MP4 \
    --output_folder /data/outputs/indoor_travel/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/indoor.yaml \
    --mode windowed --window_size 128 \
    --keyframe_interval 10 --overlap_keyframes 8 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f4f5e555-22a8-4cc9-b380-dfde5fe1c809" />

Flag-by-flag rationale:

| Flag | Why it's there |
|---|---|
| `--mode windowed --window_size 128` | Sliding-window inference is required once the sequence exceeds the ~320-frame RoPE training range; each window resets the KV cache. **`window_size` counts KV-cache slots, not actual frames** — the first `num_scale_frames` (=8) slots hold the scale frames and the remaining `128 − 8 = 120` slots hold keyframes. With `keyframe_interval = 13`, one window therefore covers `8 + 120 × 13 = 1568` actual frames. |
| `--keyframe_interval 10` | Cache only every 10th frame as a keyframe. Non-keyframes still emit per-frame predictions but don't grow the KV cache|
| `--overlap_keyframes 8` | Adjacent windows share 8 keyframes of context, resolved internally to `max(num_scale_frames, 8 × keyframe_interval) = 8 × 13 = 104` actual frames of overlap. Recommended whenever `keyframe_interval > 1`, to keep cross-window pose alignment stable. |
| `--config demo_render/config/indoor.yaml` | Seed render/scene/camera/overlay defaults from the indoor preset (short depth, tighter follow cam). Any CLI flag the user explicitly passes still overrides the YAML value. |
| `--sky_mask_dir` / `--sky_mask_visualization_dir` | Persist sky masks and their side-by-side visualizations to disk so subsequent reruns reuse them instead of re-running ONNX segmentation. (The render pipeline only consumes them when sky masking is enabled — by the YAML preset or by `--mask_sky`.) |
| `--camera_vis default` | Overlay the trajectory trail + recent-frame points on the rendered video. |
| `--keyframes_only_points` | Only unproject keyframe depth into the point cloud; non-keyframes still contribute their pose to the trajectory/frustum overlay. Keeps the cloud sparse for very long sequences. |
| `--frame_tag --frame_tag_position top_right` | Stamp a `<i> / <N> Frames` counter in the top-right corner of the MP4. |
| `--save_predictions` | Persist per-frame NPZs alongside the MP4. Useful for inspection or for re-rendering with different camera/overlay settings later. |


#### Quick Mode and Demo Reproduction

Replacing keyframe_interval = 10 with image_stride = 10 speeds up rendering. Then, uncomment the camera follow section in demo_render/config/indoor.yaml and set the birdeye's ranges to [2000, 2500] to reproduce the indoor fly-through effect shown in the demo:

<img width="3822" height="1080" alt="image" src="https://github.com/user-attachments/assets/5581d2b2-cb86-4187-a13d-46ac9a22ce99" />





https://github.com/user-attachments/assets/21b444ea-e6b6-48f0-8b34-3acad41166ac







### Worked Example — outdoor drive scene

**Dataset:** Download the example video from [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) on Hugging Face.

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/drive_frames.mp4 \
    --output_folder /data/outputs/drive/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/outdoor_drive.yaml \
    --mode windowed --window_size 128 \
    --max_non_keyframe_gap 100 --overlap_keyframes 8 \
    --image_stride 1 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

<img width="3822" height="1080" alt="image" src="https://github.com/user-attachments/assets/3c26afdb-6bb8-4d20-a7e0-f5a220382662" />


What differs from the indoor walkthrough above:

| Flag | Why it's there |
|---|---|
| `--config demo_render/config/outdoor_drive.yaml` | Seed defaults from the outdoor preset: sky masking enabled, deeper render range (`max_depth: 250`), and a follow cam tuned for vehicle trajectories with a final birdeye reveal. |
| `--image_stride 1` | Use every video frame. Increase it to subsample long or high-FPS drive footage. |
| `--max_non_keyframe_gap 100` | Upper bound on consecutive non-keyframes before a keyframe is forced. Only active with flow-based keyframe selection (`--flow_threshold > 0`); in the default fixed-interval mode it has no effect. |

The remaining flags (`--mode windowed --window_size 128`, `--overlap_keyframes 8`, sky-mask caching, overlays, `--save_predictions`) carry over unchanged from the indoor example — see the flag-by-flag table above.

### Worked Example — LingBot-World scenes

Reconstruct videos generated by LingBot-World, our world model — the same pipeline works on generated footage out of the box.

**Dataset:** Download the example videos (`lingbo_world_frames.mp4`, `lingbo_world2_frames.mp4`) from [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) on Hugging Face.

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/lingbo_world_frames.mp4 \
    --output_folder /data/outputs/lingbo_world/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/outdoor_drive.yaml \
    --mode windowed --window_size 128 \
    --max_non_keyframe_gap 100 --overlap_keyframes 8 \
    --image_stride 1 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

For the second clip, run the same command with `--video_path /data/demo_videos/lingbo_world2_frames.mp4 --output_folder /data/outputs/lingbo_world2/` (and separate `--sky_mask_dir` / `--sky_mask_visualization_dir` folders if you want to keep the cached masks apart).

All flags are identical to the [outdoor drive scene](#worked-example--outdoor-drive-scene) above — only the input video and output folder change. See the drive scene and indoor walkthrough tables for the flag-by-flag rationale.

<img width="3736" height="1080" alt="image" src="https://github.com/user-attachments/assets/1f60d505-1407-482c-9b5d-57c7145c0b7d" />

<img width="1200" height="339" alt="image" src="https://github.com/user-attachments/assets/e62bedaa-1e61-40b3-8fea-01c8a15355f0" />



### Camera Path (YAML)

The virtual camera path is described by the `camera.segments` list in the YAML preset passed via `--config`. Edit the YAML to design your own shot — no need to touch CLI flags.

Built-in presets live in `demo_render/config/`: `default.yaml`, `indoor.yaml`, `outdoor_drive.yaml`. Copy one and edit the `camera:` block.

#### YAML structure

```yaml
camera:
  fov: 60.0          # camera field of view in degrees
  transition: 30     # frames blended between adjacent segments
  segments:
    - mode: follow            # chase cam following the input trajectory
      frames: [0, 1500]       # rendered-frame range this segment covers (-1 = end)
      back_offset: 0.3        # how far behind the input camera (fraction of scene scale)
      up_offset: 0.08         # vertical lift above the input camera
      look_offset: 0.4        # how far ahead the lookat target points
      smooth_window: 30       # trajectory smoothing window in frames
    - mode: birdeye           # rise up for a top-down reveal of the whole scene
      frames: [1500, 1800]
      reveal_height_mult: 2.5 # birdeye height = scene scale × this factor
    - mode: follow            # drop back into chase cam
      frames: [1800, -1]
      back_offset: 0.3
      up_offset: 0.08
      look_offset: 0.4
```

`transition` controls how many frames are blended between adjacent segments; `frames: [0, -1]` means "the whole sequence".

#### Available modes

| `mode` | Behavior | Tunable fields |
|---|---|---|
| `follow` | Chase cam tracks the input trajectory with smooth offsets. The most cinematic option for walkthroughs. | `back_offset`, `up_offset`, `look_offset`, `smooth_window`, `scale_frames` |
| `birdeye` | Top-down reveal of the whole scene. Useful for hero / overview shots. | `reveal_height_mult` |
| `static` | Fixed eye + lookat, auto-derived from the segment's start frame. | — |
| `pivot` | Fixed eye, lookat sweeps along the trajectory. | — |

#### Single-shot YAML examples

**Pure follow** (most common):

```yaml
camera:
  fov: 60.0
  segments:
    - mode: follow
      frames: [0, -1]
      back_offset: 0.3
      up_offset: 0.08
      look_offset: 0.4
      smooth_window: 30
```

**Full birdeye** (good for overview / hero shots):

```yaml
camera:
  fov: 60.0
  segments:
    - mode: birdeye
      frames: [0, -1]
      reveal_height_mult: 2.5
```

**Follow with birdeye inserts**: just list multiple segments in order under `segments:` — adjacent segments are interpolated using `transition` frames.

> Caveat: when `--config` loads a YAML preset, passing **any** segment-shaping CLI flag (`--camera_mode`, `--back_offset`, `--up_offset`, `--look_offset`, `--smooth_window`, `--follow_scale_frames`, `--birdeye_start`, `--birdeye_duration`, `--reveal_height_mult`) discards the YAML's `segments` and rebuilds the camera path from those flags instead. To stay fully YAML-driven, don't pass any of them on the command line.

### Output files

For a given output name (e.g. `<scene>` or `<video_name>`):

| File | Description |
|------|-------------|
| `<name>_pointcloud.mp4` | Rendered point-cloud flythrough |
| `<name>_pointcloud_rgb.mp4` | Original RGB frames encoded as video |
| `<name>_pointcloud_config.yaml` | Full config snapshot of this run |
| `batch_results.json` | Per-scene success / duration summary |

## 📜 License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE.txt) file for details.

## 📖 Citation

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

## ✨ Acknowledgments

We thank Shangzhan Zhang, Jianyuan Wang, Yudong Jin, Christian Rupprecht, and Xun Cao for their helpful discussions and support.

This work builds upon several excellent open-source projects:

- [VGGT](https://github.com/facebookresearch/vggt)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Flashinfer](https://github.com/flashinfer-ai/flashinfer)

---
