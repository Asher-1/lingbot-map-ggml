# Sky-Segmentation GGUF + Latency Matrix

Two deliverables closed on 2026-09-12:

1. **A native ggml sky-segmentation runtime** (no onnxruntime / no Python in
   the inference path), converted from the official `skyseg.onnx`, in three
   quantizations.
2. **The cross-engine latency matrix** for the main reconstruction, together
   with the accuracy figures already recorded in `validation_report.md`.

## Sky segmentation: official onnx vs native ggml GGUF

The official viewer's `--mask_sky` runs `skyseg.onnx` through onnxruntime
(`lingbot_map/vis/sky_segmentation.py`). The same network now runs as a
native ggml graph:

- converter: `cpp_ggml/scripts/convert_skyseg.py skyseg.onnx out.gguf --outtype f32|f16|q8_0`
  (onnx-simplifier constant-folds the dynamic shape arithmetic, then emits
  a flat instruction list into the GGUF metadata; 43.98M params, 375 ops)
- runtime: `cpp_ggml/src/skyseg.cpp` — im2col+mul_mat conv path (q8_0 kernels
  as 2-D `{K,OC}` mul_mat operands, f16/f32 via the same operand layout),
  MaxPool, bilinear Resize (pytorch_half_pixel), Concat/Add/Sigmoid
- CLI: `lingbot-map-cli ... <skyseg.gguf>` (argv[9]) zeroes each frame's
  depth confidence for sky pixels before any output is written — both the
  streamed frames and the final LBO/LBP2 — matching the official
  `(1 - prob) > 0.1` keep rule. ggml-demo needs no onnxruntime for
  `--mask_sky` anymore.
- standalone fidelity/latency runner: `cpp_ggml/build-*/skyseg-cli`

### Mask fidelity vs onnxruntime (courthouse frame 0, 320x320)

After the preprocess alignment below (bit-exact cv2 u8 resize), the raw
sigmoid map agrees with onnxruntime to quantization-noise level. Before the
alignment the same table read max 0.619 / RMSE 0.0033 for every variant —
an engine gap dominated by the resize semantics, not by the weights
(measured with scripts/evaluate_skyseg_alignment.py; sky IoU = raw > 0.5):

| skyseg variant / backend | max abs diff | prob RMSE | sky IoU (raw > 0.5) |
| --- | ---: | ---: | ---: |
| ggml f32 CPU / Vulkan | 0.00000 | 0.000000 | 1.00000 |
| ggml f32 CUDA0 | 0.00056 | 0.000015 | 1.00000 |
| ggml f16 CPU / Vulkan | 0.00129 / 0.00082 | 0.000029 / 0.000027 | 1.00000 |
| ggml f16 CUDA0 | 0.00324 | 0.000066 | 0.99993 |
| ggml q8_0 CPU / CUDA0 / Vulkan | 0.028 / 0.024 / 0.025 | 0.0007 | 0.99951-0.99972 |

f32 on CPU and Vulkan is bit-exact against onnxruntime; CUDA shows fp
reassociation noise of the same magnitude as the f16 quantization noise.
The residual spread across quantizations is exactly the weight-format
noise (f16 ~5e-4 relative, q8_0 block dequant), two orders of magnitude
below the 1/255 u8 lattice the mask consumes.

![Mask fidelity (left) and per-frame latency (right) from the earlier same-day
measurement round, before the resize fix and the final 20-iter
re-measurement; the table above carries the final numbers](skyseg_ggml_fidelity_latency_20260912.png)

### End-to-end mask-semantics alignment (2026-09-12, revised same day)

The CLI's mask pipeline replicates the official postprocess **exactly**,
including its surprising semantics: `run_skyseg` min-max-normalizes the raw
sigmoid map, truncates to u8, resizes to the frame, and the official uint8
clip (`_mask_to_float` clips a 0..255 array to [0,1]) collapses every
non-zero value — so the official keep rule is precisely "resized u8 value
== 0".

The first measurement of this protocol still showed **99.35% pixel
agreement** over the 8 courthouse frames, with frame 7 at 97.89% and the
white tent roof being classified as sky by the ggml chain but not by
onnxruntime. Per-layer isolation (same 320x320 input into both engines)
showed the ggml network itself matches onnxruntime to rmse 1.5e-05 — the
entire gap was the **input resize**: the C++ preprocessing used a textbook
float bilinear + lround on the u8 image, while the official chain runs
cv2.resize on u8, whose fixed-point path quantizes coefficients to 11 bits
with two independent cvRounds and truncates the vertical reduction in two
shifts ((b0*(h0>>4)>>16) + (b1*(h1>>4)>>16) + 2) >> 2. Those sub-lsb input
differences (3853 pixels differ by 1 u8 level, 4 border pixels by up to 10)
are amplified by the network and then clipped by the min-max/u8 lattice
into visible mask flips.

Fix: `resize_u8_exact` in `src/skyseg.cpp` replicates the OpenCV 5.x CPU
u8 INTER_LINEAR path bit-exactly (verified 0 mismatches over 9.2M pixels
of courthouse preprocessing and 0 mismatches on the 320->frame mask
resize; probe script `scripts/evaluate_skyseg_alignment.py`, effect sheet
`scripts/plot_skyseg_effect.py`). End-to-end over the 8 courthouse frames
(official onnxruntime chain vs native ggml chain, keep-rule agreement):

| variant | CPU backend | CUDA0 | Vulkan0 |
| --- | ---: | ---: | ---: |
| ggml f32 | 100.00% | 100.00% | 100.00% |
| ggml f16 | 100.00% | 100.00% | 100.00% |
| ggml q8_0 | 99.99% | 99.99% | 99.99% |

keep-IoU is 1.0000 for f32/f16 and >= 0.9998 for q8_0 (its residual is
weight-quantization noise, not pipeline semantics). The earlier "residual
is cv2's fixed-point u8 resize vs the float resize" explanation was the
right subsystem but understated it: the resize was not merely close, it
needed to be bit-exact.

![Effect sheet over the 8-frame courthouse stream: input frame, official
onnxruntime mask, GGML f16 mask, pixel disagreement — sky tinted red, the
sampled frames all read agreement 100.00%](skyseg_effect_comparison_20260912.png)

The CLI also gained the official mask tooling: argv[10] is a mask-cache dir
(PNG per frame, {0,255}, reused across runs exactly like the official
`<folder>_sky_masks` cache — a second run skips the network entirely), and
argv[11] a visualization dir writing the official-style
original | mask | overlay panel per frame. `ggml_demo.py --mask_sky
--sky_mask_dir ... --sky_mask_visualization_dir ...` wires both through.

### Open vocabulary: no

`skyseg.onnx` (JianyuanWang/skyseg) is a fixed-class sky/non-sky U-Net, and
the official demo's `--mask_sky` uses exactly this model — there is no
open-vocabulary (text-prompted) segmentation anywhere in the official
pipeline. The GGUF variants are weight-format conversions of that same
network, so "aligned with the official demo" is the correct and complete
alignment target; there is nothing further to add.

### Skyseg latency (RTX 4090, host Intel i9-14900K, 20-iter average, ms/frame)

The executor is part of every number below: CUDA0/Vulkan0 rows run on the
NVIDIA RTX 4090, CPU rows run on the host Intel i9-14900K.

| engine | backend | ms/frame |
| --- | --- | ---: |
| onnxruntime | CPU, Intel i9-14900K (the official path) | 104.1 |
| ggml f32 | CPU (i9-14900K) | 863.3 |
| ggml f16 | CPU (i9-14900K) | 1083.0 |
| ggml q8_0 | CPU (i9-14900K) | 873.4 |
| ggml f32 | CUDA0 (RTX 4090) | 16.4 |
| ggml f16 | CUDA0 (RTX 4090) | 17.6 |
| ggml q8_0 | CUDA0 (RTX 4090) | 18.1 |
| ggml f32 | Vulkan0 (RTX 4090) | 25.9 |
| ggml f16 | Vulkan0 (RTX 4090) | 27.2 |
| ggml q8_0 | Vulkan0 (RTX 4090) | 27.9 |

Chart: `skyseg_latency_matrix_20260912.png`. GPU backends are 4-6.3x
faster than the official onnxruntime CPU path; at GPU speed the three
quantizations are indistinguishable (f16 17.6 vs q8_0 18.1 ms on CUDA0),
so f16 remains the default choice and q8_0 buys model size, not speed.

![Skyseg latency matrix (earlier same-day measurement round; the table above
carries the final 20-iter numbers)](skyseg_latency_matrix_20260912.png)

Notes: the ggml CPU path is not tuned (im2col+mul_mat per layer, single
measurement batch); the production path is the GPU backend of the running
reconstruction, where skyseg costs ~21-35 ms per frame. **Q8_0 IS usable on
Vulkan** (2026-09-12 correction of the earlier claim): the blocker was never
a missing Vulkan q8 matmul - the backend ships a native
`matmul_q8_0_q8_1` shader - but three chained v0.21 limitations: (1) the
auto-quantization of F32 activations to Q8_1 requires the
`VK_KHR_shader_integer_dot_product` extension, which reports unsupported on
this driver (`int dot: 0`), (2) with F32 activations the (Q8_0, F32) mul_mat
falls through to the generic f32 op path, whose assert rejects any
quantized operand, and (3) neither im2col nor cpy can emit Q8_1 tensors on
Vulkan, so the shader's required operand cannot be produced. Fix: Q8_0
kernels are dequantized to F16 at load time on Vulkan (host-side, one-time,
~5e-4 relative rounding bound); CUDA and CPU run the native Q8_0 path.
Measured Vulkan q8_0 (RTX 4090): 27.9 ms/frame, raw sky IoU 0.99972 vs
onnxruntime (e2e keep-rule agreement 99.99%) - statistically the same
speed and fidelity as f16.

## Reconstruction latency matrix (PyTorch vs GGUF, 8 frames, wall incl. engine load)

Measured with the same protocol as the parity rows: courthouse, native
aspect 518x294, `scale=8/window=64`, NVIDIA RTX 4090, host Intel
i9-14900K (chart:
`reconstruction_latency_matrix_20260912.png`).

| engine / format | backend | 8-frame wall (s) |
| --- | --- | ---: |
| PyTorch .pt (bf16, FlashInfer) | CUDA | 13.1 |
| GGML q8 | CUDA0 | 5.0 |
| GGML f16 | CUDA0 | 5.6 |
| GGML f32 | CUDA0 | 6.7 |
| GGML q8 | Vulkan0 | 5.4 |
| GGML f16 | Vulkan0 | 6.1 |

![Reconstruction latency matrix: official PyTorch vs GGUF formats x backends,
8-frame streaming wall incl. engine load (RTX 4090)](reconstruction_latency_matrix_20260912.png)

The GGML runtime starts and finishes faster than the PyTorch pipeline on
this protocol (PyTorch pays torch import + 4.4 GB checkpoint load); raw
per-frame inference with FlashInfer remains faster on the PyTorch side for
long streams, which is the trade documented in `MODEL_CARD.md`.

## End-to-end C++ closure

With skyseg native, the GGML engine covers the whole outdoor-reconstruction
pipeline in C++: preprocessing (official crop, bit-identical), streaming
reconstruction with the persistent F16 KV cache, per-frame streaming output,
and sky-masked confidence — `run_gui.sh --engine ggml --mask_sky` needs no
Python inference stack beyond the viewer itself.
