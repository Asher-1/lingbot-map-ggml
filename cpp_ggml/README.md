# LingBot-Map C++ GGML

This directory pins GGML `v0.21.0` as `third_party/ggml`. Every local GGML source
change is consolidated in exactly one patch,
`third_party/patches/0001-lingbot-ggml-v021.patch`. CMake replays or
reverse-validates that patch at configure time. Do not edit the submodule
directly.

For a reproducible newcomer path, use `bash cpp_ggml/scripts/run_e2e.sh cuda q8`
or `bash cpp_ggml/scripts/run_e2e.sh vulkan f16`. It configures, builds, runs
the official sequence at the native aspect (518×294, the identity output of the
official `load_and_preprocess_images(mode="crop", image_size=518)` for these
frames), exports PLY/image evidence, and enforces strict parity.
See `models/MODEL_CARD.md` for model download and deployment scope.

## Quick Start: Clone To Full Reconstruction

This CUDA 11.8 / RTX 30-series path runs the real streaming C++ GCT graph over
every official `courthouse` frame, generates an independent PyTorch reference
from the same GGUF weights, and writes parity/reconstruction artifacts. The
one-command path assumes q8/f16 GGUF files are already downloaded; it never
requires cuDNN.

```bash
git clone --recurse-submodules git@github.com:Asher-1/lingbot-map-ggml.git
cd lingbot-map-ggml
git submodule update --init --recursive

# Python is used only for conversion, the independent reference, and plots.
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e . gguf matplotlib
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Download released GGUF files. Original .pt files, when conversion is needed,
# belong in cpp_ggml/models/pytorch/.
python3 -m pip install "huggingface_hub[cli]"
huggingface-cli download Asher-1/lingbot-map-gguf \
  --local-dir cpp_ggml/models/gguf

# Optional: make a GGUF locally from the original checkpoint.
python3 cpp_ggml/scripts/convert_lingbot.py \
  cpp_ggml/models/pytorch/lingbot-map.pt \
  cpp_ggml/models/gguf/lingbot-map-q8.gguf --outtype q8

cmake -S cpp_ggml -B cpp_ggml/build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DLINGBOT_GGML_CUDA=ON -DLINGBOT_GGML_CUDA_ARCHITECTURES=86-real \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build cpp_ggml/build-cuda --parallel 6

# Builds (and performs CMake patch replay), runs all 286 official frames at
# the native aspect, validates elementwise output, and writes CSV, PLY, and
# comparison images.
bash cpp_ggml/scripts/run_e2e.sh cuda q8
```

For f16 use `bash cpp_ggml/scripts/run_e2e.sh cuda f16`. Vulkan uses
`bash cpp_ggml/scripts/run_e2e.sh vulkan q8`; omit the final frame argument
only for the complete 286-frame gate. The command writes a CSV summary,
depth/trajectory image, point-cloud comparison image, two colored PLYs, LBO1
raw outputs, and C++ LBP2 pose/intrinsics outputs. It exits nonzero when the
scene-level gate fails. Use the strict parity command below for the elementwise
gate; reconstruction RMSE alone is not a parity claim.

## One-Command 3D GUI (both engines)

The official reconstruction viewer can be driven by either engine with one
command — both share the same preprocessing (official crop to `--image_size`,
518 by default) and the same streaming profile (scale=8/window=64, persistent
F16 KV cache), so the two reconstructions are directly comparable:

```bash
bash run_gui.sh                                        # ggml, courthouse, all frames
bash run_gui.sh --engine ggml --frames 40              # quick look
bash run_gui.sh --engine pytorch --image_folder example/loop
bash run_gui.sh --engine ggml --backend CUDA0 --build cpp_ggml/build-cuda

With the default settings the GGML engine opens a live viser view
immediately and grows point clouds + camera frustums frame-by-frame
during the native streaming run; the full official viewer takes over
on the same port when inference completes.
```

The script probes system/conda/venv interpreters for the viewer dependencies
(`pip install viser trimesh`), auto-installs them when only those are missing,
auto-builds the C++ runtime on first run, and opens `http://localhost:8080`.
The `cpp_ggml/scripts/run_gui.sh` entry point is a thin forwarder to the root
script with `--engine ggml` pre-selected.

Useful env overrides: `PYTHON`, `GGML_MODEL`, `GGML_BUILD`, `LINGBOT_CKPT`.

## Build

```bash
git submodule update --init --recursive
cmake -S cpp_ggml -B cpp_ggml/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_ggml/build --parallel 6
```

Enable optional backends with `-DLINGBOT_GGML_CUDA=ON` or
`-DLINGBOT_GGML_VULKAN=ON`. CUDA defaults to cuBLAS f32 accumulation for q8
parity, with TF32 disabled by the consolidated patch so its reduction semantics
match the PyTorch reference. The same patch preserves f32 activation and
convolution accumulation on Vulkan. CMake requires exactly one patch, applies
it to a clean submodule, or verifies that the same patch is already applied.
Do not edit the GGML submodule in place.
The C++ Vulkan backend runs every mode on the exact scalar F32 paths by
default: strict mode disables cooperative-matrix, integer-dot, and fp16
Vulkan arithmetic entirely, and flash mode keeps the GEMMs scalar while the
flash-attention dispatch uses the coopmat1 f32acc pipeline (P hi/lo split +
F32 output chain, see `AGENTS.md`). `LINGBOT_VULKAN_FAST=1` opts out of the
blanket disables for a separately labelled non-parity throughput experiment.
The independent PyTorch reference and benchmark explicitly disable cuDNN and
cuBLAS TF32 by default. The C++ runtime links GGML, CUDA runtime/cuBLAS, or
Vulkan only; it neither finds nor links cuDNN. This keeps the validation
pipeline on standard PyTorch CUDA operators plus GGML/cuBLAS, matching GGML's
f32 convolution and reduction contract. Pass `--allow-tf32` only when
collecting a separately labelled throughput baseline, never for parity.
For RTX 30-series builds, constrain compilation to the installed GPU:

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cuda \
  -DLINGBOT_GGML_CUDA=ON -DLINGBOT_GGML_CUDA_ARCHITECTURES=86-real \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

MMQ is mutually exclusive with the default cuBLAS path and intentionally
trades numerical parity for throughput. Enable it only for performance
experiments:

```bash
cmake -S cpp_ggml -B cpp_ggml/build-mmq -DCMAKE_BUILD_TYPE=Release \
  -DLINGBOT_GGML_CUDA=ON -DLINGBOT_GGML_CUDA_ARCHITECTURES=86-real \
  -DLINGBOT_GGML_CUDA_FORCE_CUBLAS=OFF -DLINGBOT_GGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build cpp_ggml/build-mmq --parallel 6
```

## Model assets

Put the original checkpoint in `models/pytorch` and generated files in
`models/gguf`:

```bash
python3 cpp_ggml/scripts/convert_lingbot.py \
  cpp_ggml/models/pytorch/lingbot-map.pt \
  cpp_ggml/models/gguf/lingbot-map-f16.gguf --outtype f16
# long-sequence checkpoint: identical architecture, same command shape
python3 cpp_ggml/scripts/convert_lingbot.py \
  cpp_ggml/models/pytorch/lingbot-map-long.pt \
  cpp_ggml/models/gguf/lingbot-map-long-f16.gguf --outtype f16
```

The `lingbot-map-long-*` GGUFs (f32/f16/q8) convert the upstream
`lingbot-map-long.pt` checkpoint — architecture-identical to the balanced
models, so every graph option and tool applies unchanged; their validation
status and provenance (source sha256, smoke-gate numbers) live in
`models/MODEL_CARD.md`. `run_e2e.sh` and `run_benchmarks.py` take a `long`
variant argument (`run_e2e.sh vulkan f16 40 long`, `run_benchmarks.py
--variant long`).

The converter supports `f32`, `f16`, `q8`, and `q4`. Quantized tensors are
limited to representable 2-D matrices; norms, biases, and convolution kernels
remain floating point. Conversion fails for a missing or malformed checkpoint.
To evaluate a deployment format, use its generated GGUF as
`--reference-gguf`, so weight-format loss is not reported as a graph mismatch.
f32 and q4 are not currently validated on the 12 GiB target; a successful
conversion alone is not backend validation.

## Runtime contract

`lingbot-map-cli` accepts a raw float32 `[1,F,3,H,W]` buffer and a GGUF file;
the optional final `frames` argument overrides `F`. The release profile
processes frames one-at-a-time to bound graph memory while preserving the
global-attention and CameraCausalHead KV state for the active stream.
Pass a sixth argument to dump a binary result, then compare it with a PyTorch
`.npz` reference using `scripts/compare_parity.py`.
`lingbot-map-bench` emits machine-readable `RESULT` lines. The current C++
runner builds the GCT data path with ImageNet normalization, DINO patch
embedding, alternating frame/global transformer blocks, RoPE, persistent causal
K/V state, iterative CameraCausalHead refinement, and the complete four-scale
DPT projection, resize, scratch, RefineNet, and output-convolution chain.
CameraHead q8 linear weights are dequantized once per graph into shared F32
nodes before its iterative AdaLN gate; this prevents q8 reduction error from
being amplified across pose refinement while DINO and DPT remain q8.
`model::reset_cache()` starts a new stream. CUDA and Vulkan builds upload GGUF
tensors into backend buffers and use the same graph contract; actual device
availability and kernel coverage are reported by the selected backend at
runtime. Numeric parity and scene reconstruction must be measured with the
released checkpoint and the independent reference runner below.

## Measurements

```bash
python3 cpp_ggml/scripts/run_benchmarks.py --build cpp_ggml/build
python3 cpp_ggml/scripts/plot_reconstruction.py reconstruction.csv

# Official bundled sequence: every courthouse frame at the native aspect
# (518x294), independent PyTorch/q8 parity, all-frame colored PLYs, and a
# point-cloud comparison image.
python3 cpp_ggml/scripts/run_reconstruction.py \
  --build cpp_ggml/build-cuda --backend CUDA0 \
  --model cpp_ggml/models/gguf/lingbot-map-q8.gguf \
  --reference-gguf cpp_ggml/models/gguf/lingbot-map-q8.gguf \
  --scene example/courthouse --frames 286 \
  --out cpp_ggml/benchmarks/reconstruction_compare_q8_courthouse_full.csv \
  --effect-out cpp_ggml/benchmarks/reconstruction_effect_q8_courthouse_full.png \
  --full-export-prefix cpp_ggml/benchmarks/reconstruction_full_q8_courthouse
```

`run_benchmarks.py` records only successfully measured backend/dtype rows.
`plot_benchmarks.py` refuses to produce a latency matrix until CPU/CUDA/Vulkan
and f32/f16/q8/q4 rows all exist. `plot_reconstruction.py` requires measured
scene-level ATE and depth RMSE for both engines; no placeholder data is created.

## Official Benchmark Interface

The root `benchmark/` package has a native `lingbot_map_ggml` method config.
It feeds BSS-resized RGB tensors to the C++ CLI and consumes the C++ LBO1 depth
plus LBP2 C2W/intrinsics sidecar; it does not import the PyTorch model or
reimplement pose decoding. Run it from the benchmark directory after preparing
the target BSS dataset:

```bash
cd benchmark
# Select lingbot_map_ggml in the copied dataset config's methods list, then:
python run.py --config configs/<dataset>.yaml --debug --force
```

## Current Evidence

The full parity table (eight rows at the native aspect plus three stretched
profiles, both backends, q8 GGUF, F16 KV cache, 286 frames each) lives in
`benchmarks/validation_report.md` under "Persistent F16 KV Cache". The
headline native-aspect rows:

| Backend | Pose RMSE | Depth RMSE | pose viol | depth viol | worst rel depth err | max center dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA0 | 6.14e-05 | 9.75e-05 | 0 / 2574 | 0.0100% | 2.09% | 6.2e-04 m |
| Vulkan0 | 3.89e-05 | 9.71e-05 | 0 / 2574 | 0.0096% | 2.02% | 4.7e-04 m |

The two backends also agree directly with each other (pose RMSE 4.26e-05,
depth RMSE 2.78e-05, worst single-pixel depth difference 5.9e-03 at a
discontinuity), and both sidecars pass the postprocess gate at this profile
(`compare_postprocess.py`: c2w max_abs 1.19e-06 CUDA / 1.43e-06 Vulkan).
Camera-center deviations are measured over a ~4 m walk-around trajectory.

Against the official fp32/bf16 checkpoint (not just the decoded-GGUF
mirror), the engine is fully aligned: GGML q8 and PyTorch decoding the same
q8 GGUF deviate from the official checkpoint by identical amounts
(pose `2.19e-04` vs `2.09e-04`, depth `1.56e-03` both), and the f16 GGUF
closes the remaining weight-format gap to pose `1.72e-04` / depth
`4.72e-04` over the full 286-frame stream. Preprocessing is bit-identical
(`max_abs_diff=0` vs the official loader). Full matrix:
`benchmarks/validation_report.md`, section "End-to-End Alignment Against
the Official PyTorch Pipeline".

**Fast mode:** `--kv-f16 flash` streams through the FA kernels with
F32-effective PV numerics (F32 VKQ accumulation + P hi/lo split on CUDA; on
Vulkan the coopmat1 FA with the same P hi/lo split and an F32 output chain,
GEMMs kept scalar) and an F32 camera-trunk cache, holding the same 1e-4-class
parity at FlashInfer speed: 286-frame native-aspect rows — CUDA f16 1.39e-04 /
4.55e-04 at 2m11s, f32 1.29e-04 / 1.22e-04, q8 1.33e-03 / 3.17e-03 (the
documented q8 weight cost); Vulkan f16 7.27e-05 / 1.06e-04 at 2m39s, f32
7.25e-05 / 1.35e-04 at 2m33s, q8 1.34e-03 / 3.16e-03 at 2m30s. See
`benchmarks/validation_report.md` and `AGENTS.md`.

Earlier F32-cache rows (q8/f16 at 518×518 `scale=1/window=4`) are recorded
in `benchmarks/validation_report.md` with the same magnitude of pose/depth
RMSE (3.5e-06 / 2.75e-06), but their per-frame renders were superseded by
the 2026-09-09 pose-fix batch.

Reference figures (artifacts live in `benchmarks/`; full tables and context
in `benchmarks/validation_report.md`):

![286-frame fused point clouds, PyTorch reference vs GGML f16 KV cache on
CUDA at the native 518x294
aspect](benchmarks/reconstruction_full_f16cache_cuda-native_courthouse_20260909_comparison.png)

![Per-frame depth effect sheet: RGB, reference depth, GGML depth, abs-diff
at four sampled frames, plus the camera
trajectory](benchmarks/reconstruction_effect_f16cache_cuda_courthouse_native_full_20260909.png)

![Outdoor end-to-end with native C++ sky segmentation: GGML f16 vs the
official PyTorch pipeline — point clouds, trajectory, sky
panels](benchmarks/outdoor_e2e_comparison_20260912.png)

## Parity And Stream Diagnostics

`run_reconstruction.py` generates the reference internally. For a stored C++
LBO1 result and its matching reference, use the strict elementwise gate:

```bash
python3 cpp_ggml/scripts/compare_parity.py cpp_output.lbo pytorch_reference.npz \
  --atol 0.001 --rtol 0.001

# Verify C++ C2W and intrinsics decode against the official Python utility.
python3 cpp_ggml/scripts/compare_postprocess.py cpp_output.lbo --height 294 --width 518
```

To inspect one real streaming frame without serializing every intermediate
tensor, select its zero-based inference index. The C++ dump includes every
DINO/frame/global attention block plus DPT projection, RefineNet, and
output-logit boundaries. The PyTorch command retains matching DPT hooks from
the final streamed frame. Use an input sequence ending at the selected frame, so frame
272 requires the first 273 official frames. The full-reconstruction command
above leaves its normalized official input at `/tmp/lingbot_scene.npy` and
`/tmp/lingbot_scene.bin`; create the required prefix once:

```bash
python3 - <<'PY'
import numpy as np
frames = np.load('/tmp/lingbot_scene.npy')[:, :273].astype(np.float32, copy=False)
np.save('/tmp/frames_0000_0272.npy', frames)
frames.tofile('/tmp/frames_0000_0272.bin')
PY
```

The stage-dump hooks are CLI flags now (`--dump-stages <dir>`,
`--dump-at N`, `--dump-internal`, `--dump-block`, `--dump-camera-stages`);
`LINGBOT_KV_CACHE_F16` is still honored by `ggml_demo.py` for backward
compatibility, but the canonical mode interface is the CLI flag
`--kv-f16 <none|strict|flash>`.

```bash
cpp_ggml/build-cuda/lingbot-map-cli \
  cpp_ggml/models/gguf/lingbot-map-q8.gguf /tmp/frames_0000_0272.bin CUDA0 \
  294 518 /tmp/cpp-272.lbo 273 --dump-stages /tmp/cpp-stage --dump-at 272

python3 cpp_ggml/scripts/run_pytorch_reference.py \
  cpp_ggml/models/pytorch/lingbot-map.pt /tmp/frames_0000_0272.npy /tmp/torch-272.npz \
  --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 \
  --gguf cpp_ggml/models/gguf/lingbot-map-q8.gguf --dump-stages --dump-dpt-stages

python3 cpp_ggml/scripts/compare_stage_dumps.py /tmp/cpp-stage /tmp/torch-272.npz \
  --frame 272 | tee cpp_ggml/benchmarks/stage_parity_q8_courthouse_frame272.tsv
```

`LINGBOT_DUMP_AT` affects diagnostics only. It neither resets nor changes the
global/camera KV cache, so the selected frame retains its true streaming state.
The comparison report stops at the first count mismatch; this makes an
incorrect boundary mapping explicit instead of comparing reshaped tensors.
