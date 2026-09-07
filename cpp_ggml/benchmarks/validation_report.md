# Validation Report — Final State

Date: 2026-09-14 (consolidated; earlier intermediate experiment logs live in
git history and the process knowledge is distilled into the root `AGENTS.md`).
All rows: official 286-frame `courthouse` stream, NVIDIA RTX 4090 (24 GiB) +
Intel i9-14900K host, native aspect 518x294, `scale=8/window=64` unless noted.
Every number regenerates via `cpp_ggml/scripts/run_reconstruction.py` /
`run_pytorch_reference.py` / `run_e2e.sh`.

## 1. Mode Contract

| mode | selection | attention path | 286-frame wall (f16) | accuracy vs official fp32 | purpose |
| --- | --- | --- | ---: | --- | --- |
| **strict** (default) | `--kv-f16 strict` | hand-written F32 attention in exact query chunks, F16 cache upload, device-resident payloads | CUDA 6m47s / Vulkan 7m05s | 1.423e-04 / 4.524e-04 | bit-level validation route |
| **flash** | `--kv-f16 flash` | tensor-core FA with F32-effective PV numerics (see `AGENTS.md`), GEMMs scalar, F32 camera-trunk cache | **CUDA 2m11s / Vulkan 2m39s** | 1.393e-04 / 4.546e-04 | FlashInfer-class speed at parity accuracy |
| PyTorch FlashInfer reference | — | — | 2m14s | — | the speed bar |

Both modes hold the same 1e-4-class parity; the scale pass always keeps exact
F32 attention. Reference chains: *graph parity* = decoded-GGUF
cache-precision-aware PyTorch mirror (`run_pytorch_reference.py --gguf
--streaming`); *end-to-end* = official fp32 `.pt` checkpoint.

## 2. Flash Route — Final Matrix (the headline optimization)

| run | wall | pose vs official fp32 | depth vs official fp32 | pose vs mirror | depth vs mirror |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA flash, f16 GGUF | 2m11.0s | 1.393e-04 | 4.546e-04 | 7.70e-05 | 1.15e-04 |
| CUDA flash, f32 GGUF | 2m10.7s | 1.287e-04 | 1.218e-04 | — | — |
| CUDA flash, q8 GGUF | 2m27.5s | 1.334e-03 | 3.165e-03 | — | — |
| Vulkan flash, f16 GGUF | 2m39.2s | 1.842e-04 | 4.499e-04 | 7.27e-05 | 1.06e-04 |
| Vulkan flash, f32 GGUF | 2m33.2s | **7.246e-05** | 1.349e-04 | — | — |
| Vulkan flash, q8 GGUF | 2m30.1s | 1.340e-03 | 3.164e-03 | — | — |
| Vulkan strict, f16 GGUF | 7m05.3s | 1.424e-04 | 4.525e-04 | — | — |

Readings: (1) the f32-GGUF rows are the deepest graph-parity evidence —
F32-exact weights over the full stream land at 1e-04-class pose on both
backends, so the flash graph's own accumulated deviation equals the strict
mode's; (2) the q8 rows match the strict q8 rows (1.31e-03 / 3.16e-03) to
three digits — the flash route adds nothing beyond the documented q8 weight
cost; (3) per-segment RMSE does not compound with stream length (Vulkan f16
segments 8–39: 1.3e-05, 200–285: 8.3e-05). Vulkan flash runs the whole matrix
2m30–2m39s, within 1.21x of CUDA.

## 3. Strict-Mode Parity Rows

Graph parity (native C++ vs the same-GGUF decoded PyTorch reference) across
profiles and backends, default mode:

| Profile | GGUF | Backend | Pose RMSE | Depth RMSE |
| --- | --- | --- | ---: | ---: |
| scale=8/window=64, 518x294 (native) | q8 | CUDA0 | 6.14e-05 | 9.75e-05 |
| scale=8/window=64, 518x294 (native) | q8 | Vulkan0 | 3.89e-05 | 9.71e-05 |
| scale=1/window=4, 518x518 | q8 | CUDA0 | 1.07e-04 | 5.67e-05 |
| scale=8/window=64, 392x392 | q8 | CUDA0 | 9.18e-04 | 8.27e-05 |
| scale=8/window=64, 518x378 | q8 | CUDA0 | 1.66e-04 | 7.03e-05 |
| scale=1/window=4, 518x518 | q8 | Vulkan0 | 1.11e-04 | 5.76e-05 |

Native-aspect headline row (also the reconstruction-quality row):

| Backend | Pose RMSE | Depth RMSE | pose viol (1e-3) | depth viol | worst rel depth err | max center dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA0 | 6.14e-05 | 9.75e-05 | 0 / 2574 | 0.0100% | 2.09% | 6.2e-04 m |
| Vulkan0 | 3.89e-05 | 9.71e-05 | 0 / 2574 | 0.0096% | 2.02% | 4.7e-04 m |

The two backends agree directly (pose RMSE 4.26e-05, depth RMSE 2.78e-05);
camera-center deviations are measured over a ~4 m walk-around trajectory;
depth violations concentrate at discontinuities (worst relative error <= 2.1%).

## 4. Engine Alignment vs the Official Pipeline

| Comparison | pose RMSE | depth RMSE | Reading |
| --- | ---: | ---: | --- |
| GGML q8 CUDA vs q8 GGUF decoded in PyTorch (8 frames) | 1.23e-04 | 5.37e-05 | graph parity |
| q8-decoded PyTorch vs official checkpoint (8 frames) | 2.09e-04 | 1.56e-03 | pure q8 weight loss |
| GGML q8 CUDA vs official checkpoint (8 frames) | 2.19e-04 | 1.56e-03 | = parity + q8 loss (depth agrees to 4 digits — the engine is fully aligned) |
| **GGML f16 vs official checkpoint, full 286-frame stream (CUDA)** | **1.72e-04** | **4.72e-04** | end-to-end headline (strict mode) |
| GGML f16 vs official checkpoint, full 286-frame stream (Vulkan) | 1.75e-04 | 4.73e-04 | backend-consistent |
| GGML f32 vs official checkpoint (8 frames) | 1.26e-04 | 2.62e-06 | depth at bit-alignment level |
| GGML q8 vs official checkpoint, full 286-frame stream | 1.31e-03 | 3.16e-03 | q8 = memory default, not the full-alignment format |

Preprocessing is bit-identical to the official loader (`max_abs_diff=0` on
courthouse). All strict-row deviations sit two orders of magnitude below the
official bf16 deployment's own self-noise (9.7e-03 pose / 1.3e-02 depth).

Weight-format cost (isolated, 286 frames, native aspect): q8 adds
1.31e-03 / 3.16e-03 vs the fp32 checkpoint — stationary per-frame
quantization noise with zero slope over the stream, translation channels
dominating; halving the injection (f16 weights) halves the RMSE at every
sequence length.

## 5. Reconstruction-Quality Evidence

286-frame fused clouds unprojected through the official
`pose_encoding_to_extri_intri` + `unproject_depth_map_to_point_map` path,
`depth_conf > 1.5` (the `demo.py` default); regenerable with
`bash cpp_ggml/scripts/export_f16cache_reconstructions.sh`.

![PyTorch reference vs GGML, 286 frames at the native 518x294 aspect —
structurally indistinguishable fused
clouds](reconstruction_full_f16cache_cuda-native_courthouse_20260909_comparison.png)

![The same comparison at the square 518x518 release
profile](reconstruction_full_f16cache_cuda-518_courthouse_20260909_comparison.png)

![Camera centers, top-down and side views: PyTorch reference vs CUDA0 vs
Vulkan0 — curves overlap at plot scale and close the walk-around
loop](reconstruction_trajectory_f16cache_courthouse_native_20260909.png)

![RGB / reference depth / GGML depth / abs-diff at frames 0/95/190/285, plus
the origin-centered camera
trajectory](reconstruction_effect_f16cache_cuda_courthouse_native_full_20260909.png)

Outdoor chain (40 frames, f16 + native skyseg, both sky-masked): GGML C++
vs the official PyTorch pipeline pose RMSE **9.01e-05**.

![Outdoor end-to-end: point clouds, top-down trajectory, per-frame sky
panels](outdoor_e2e_comparison_20260912.png)

## 6. Memory and Latency (final state)

Peak VRAM, 518x378 streaming steady state (RTX 4090):

| Cache mode | Peak VRAM | Status |
| --- | ---: | --- |
| F32 cache (baseline) | 21.9 GiB | Not viable (single-frame graph + cudaMalloc fail) |
| strict (F16 cache, device-resident) | 20.2 GiB (scale-pass graph) → 14.3 GiB steady | Full 286-frame stream, both backends, all profiles |
| flash (F16 cache + tensor-core FA) | 14.3 GiB | Full 286-frame stream, both backends, all formats |

The 8-frame scale-pass graph reservation is the hard capacity floor
(weights format cannot change it): 518x518 = 28.5 GB (Vulkan) / fails 12 GiB;
448x448 = 23.1 GB Vulkan (fails 24 GiB there, passes CUDA); 518x378 = 22.4 GB
Vulkan / 11.5 GB CUDA; native 518x294 and the release `scale=1/window=4`
profile fit the tested hardware. Non-square grids run through the native DINO
positional interpolation (verified vs CUDA reference, `max_abs=1.9e-07`).

Wall-clock summary (286 frames, f16 GGUF unless noted):

| backend | strict | flash | PyTorch FlashInfer |
| --- | ---: | ---: | ---: |
| CUDA | 6m47s | **2m11s** | 2m14s |
| Vulkan | 7m05s | **2m39s** | — |

8-frame cross-engine wall incl. load (chart below): PyTorch .pt CUDA 13.1 s;
GGML q8/f16/f32 CUDA 5.0/5.6/6.7 s; GGML q8/f16 Vulkan 5.4/6.1 s.

![Reconstruction latency matrix: official PyTorch vs GGUF formats x backends,
8-frame wall incl. engine load (RTX 4090)](reconstruction_latency_matrix_20260912.png)

## 7. Companion: Native Sky Segmentation

Official `--mask_sky` matched by a native ggml runtime (no onnxruntime), with
the official mask postprocess and a bit-exact OpenCV u8 resize replication.

| skyseg variant | keep-rule agreement (8 frames) | keep-mask IoU | latency (CUDA0, ms/frame) |
| --- | ---: | ---: | ---: |
| ggml f32 | 100.00% | 1.0000 | 16.4 |
| ggml f16 | 100.00% | 1.0000 | 17.6 |
| ggml q8_0 | 99.99% | >= 0.9998 | 18.1 |

![Sky panels over the 8-frame courthouse stream: input frame, official
onnxruntime mask, GGML f16 mask, pixel disagreement](skyseg_effect_comparison_20260912.png)

Details: `skyseg_ggml.md`. Fixed-class sky/non-sky network — official-demo
parity is the complete target.

## 8. Route History (one line per landed optimization)

Full engineering detail, verification protocol and pitfalls:
root **`AGENTS.md`**.

1. **Streaming base** — persistent F16 KV cache + exact query-chunk attention:
   parity across all profiles/backends; the [S_k, S_q, heads] score tensor
   eliminated (4.7 GiB/layer would not fit).
2. **Device-resident KV cache** — fixed-capacity on-device payloads replace
   the host sidecar: 50 min → **6m47s** (4.4x), GPU util 15 → 100%,
   bit-identical over the full stream (all 214 evictions covered).
3. **CUDA flash parity** — noise attribution showed the fattn-mma half2 VKQ
   accumulator + P F16 rounding were the only parity blockers: F32 VKQ
   accumulation, P hi/lo split, np==1 F32 direct-write, F32 camera-trunk
   cache → flash 1.94e-02 → **7.70e-05** pose at 2m11s.
4. **Vulkan flash parity + speed** — coopmat1 FA (f32acc) + the same P hi/lo
   split + F32 output chain (`FA_F32ACC`-gated shaders), GEMMs kept scalar:
   4m29s → **2m39s** at 7.27e-05 pose. coopmat2 (TF32-class) and coopmat
   GEMMs (F16 activation rounding) are rejected for parity.
5. **Explicit options** — all `LINGBOT_*` environment switches replaced by
   `lingbot::model_options` + CLI flags; refactor verified bit-identical.

## 9. Open Follow-ups

1. **Upstream submission of the consolidated ggml patch**
   (`third_party/patches/0001-lingbot-ggml-v021.patch`, 10 files): CUDA
   flash-parity kernel work (Turing-gated), the Vulkan coopmat scalar-F32
   conv pipeline fix (empty-pipeline SIGFPE reproduces on any F32-conv
   workload), the `GGML_VK_FA_COOPMAT_ONLY` hook, and the gallocr re-reserve
   repro. Ready to split into upstreamable pieces when filed.
2. **q8 positioning** (no action unless requirements change): memory-saving
   default, not the parity format — the f16 ladder argument rules out a
   mixed-precision q8 variant.
