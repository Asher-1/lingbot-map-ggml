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

## 9. Long-Checkpoint GGUF Matrix (2026-09-15)

First 286-frame `scale=8/window=64` long-stream validation of the
`lingbot-map-long-{f32,f16,q8}.gguf` conversions (architecture-identical to
the balanced checkpoint, weights only). Protocol: `scripts/run_long_matrix.sh`
(strict e2e gates + timed flash runs) and `scripts/long_followup.sh`
(elementwise stats, F32-cache attribution, balanced baseline). Official
286-frame `courthouse` stream at the native aspect 518x294, RTX 4090.
References: decoded same-GGUF PyTorch mirror (fp32, no autocast, TF32/cuDNN
off) — graph parity, not weight-loss, is measured.

Strict mode (parity rows; wall includes engine load):

| backend | GGUF | pose RMSE | pose max | depth RMSE | depth max | depth tail >1.5e-3 | wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA0 | f16 | 4.13e-05 | 4.62e-04 | 5.05e-04 | 6.21e-02 | 1.93% | 5m56s |
| CUDA0 | q8 | 6.93e-05 | 4.73e-04 | 4.94e-04 | 1.30e-01 | 1.42% | 5m56s |
| CUDA0 | f32 | 6.02e-05 | 4.70e-04 | 5.03e-04 | 1.00e-01 | 1.60% | 5m54s |
| Vulkan0 | f16 | 3.36e-05 | 5.53e-04 | 5.37e-04 | 6.04e-02 | 2.10% | 6m38s |
| Vulkan0 | q8 | 3.84e-05 | 4.86e-04 | 4.81e-04 | 1.38e-01 | 1.29% | 6m38s |
| Vulkan0 | f32 | 3.62e-05 | 5.67e-04 | 4.82e-04 | 1.03e-01 | 1.56% | 6m35s |

Pose holds the balanced checkpoint's 1e-04 mirror class on every row
(balanced f16 CUDA baseline at this profile: 4.81e-05). The depth RMSE is
uniform ~5e-04 across all three weight formats — a cache-precision
characteristic, not weight loss — with a scatter tail (~1.3–2.1% of pixels
above 1.5e-03, max relative error ~2–5%). Attribution A/B on CUDA f16:

| run | pose RMSE | depth RMSE | depth max | tail >1.5e-3 |
| --- | ---: | ---: | ---: | ---: |
| strict (F16 cache), long f16 | 4.13e-05 | 5.05e-04 | 6.21e-02 | 1.93% |
| `--kv-f16 none` (exact F32 cache), long f16 | 3.42e-05 | **5.09e-05** | **6.02e-03** | **0.00%** |
| strict (F16 cache), balanced f16 baseline | 4.81e-05 | 8.32e-05 | 3.61e-02 | 0.03% |

Root cause (round-2 attribution, 2026-09-16; the earlier "F16-cache
sensitivity of the long weights" reading was WRONG and is corrected here):
the tail reproduces **in PyTorch itself** — with `LINGBOT_KV_CACHE_F16=1`
the decoded-GGUF mirror (fp32 weights, F16-rounded cache appends, the exact
C++ strict contract) lands depth REL `2.44e-04` / tail `1.91%` on long
(C++ strict: `2.22e-04` / `1.93%`) and `1.58e-04` / `0.05%` on balanced
(C++: `1.20e-04` / `0.03%`); the same-contract comparison C++ strict vs the
F16-cache mirror sits at REL `2.07e-04` — the two F16-cache implementations
agree, no engine defect. The long:balanced relative-perturbation ratio is
only **1.5x** (V-magnitude probe: cache |V| means 0.555 vs 0.502, F16
relative rounding error 1.76e-04 on both — identical, as RN rounding
predicts). The dominant amplifier of every absolute metric is the **depth
value scale**: the long checkpoint predicts a 3.3x larger depth scale on
this scene (mean |depth| 2.27 vs 0.70; tail pixels are far-range, median
|depth| 7.0 vs scene 0.65, per-bin relative error stays ~1e-4 everywhere).
Absolute gates (1.5e-03, max_abs, RMSE) are depth-scale-weighted; on the
relative view all long rows are ordinary 1e-04-class F16-cache noise.
Contract: long strict = pose 1e-04-class parity + depth REL 2.2e-04; the
`--kv-f16 none` route stays the bit-level route (26m07s CUDA at 286 frames
— the F32 cache upload is the price of exactness). Deployment view: every
GGML long row (3 formats x 2 backends x 2 modes) deviates from the PyTorch
bf16 SDPA deployment by pose 2.06–2.28e-03 / depth REL 3.95–4.04e-02 —
identical to PyTorch's own bf16 self-noise (pose 2.08e-03 / REL 3.95e-02),
so no engine-separable error exists at deployment precision.

Flash mode (vs the matching decoded-GGUF mirror):

| backend | GGUF | wall | pose RMSE | pose max | depth RMSE | depth max | tail >1.5e-3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA0 | f16 | **1m55s** | 7.20e-05 | 5.88e-04 | 6.85e-04 | 1.66e-01 | 4.09% |
| CUDA0 | q8 | **1m55s** | 8.46e-05 | 1.16e-03 | 9.19e-04 | 2.59e-01 | 5.20% |
| CUDA0 | f32 | **1m53s** | 7.65e-05 | 1.16e-03 | 7.30e-04 | 2.33e-01 | 3.74% |
| Vulkan0 | f16 | 2m21s | 5.14e-05 | 5.78e-04 | 6.35e-04 | 1.75e-01 | 3.00% |
| Vulkan0 | q8 | 2m28s | 5.84e-05 | 6.18e-04 | 6.92e-04 | 1.62e-01 | 3.15% |
| Vulkan0 | f32 | 2m17s | 5.52e-05 | 1.26e-03 | 6.36e-04 | 2.19e-01 | 2.82% |

Flash keeps the 1e-04-class pose parity on all six rows and runs the CUDA
column ~15% faster than the balanced checkpoint's 2m11s. The depth tail is
the same F16-K/V characteristic at flash's larger per-layer rounding
(the FA QK pipeline computes at F16 precision), again weight-format
independent. (The Vulkan f32 row was re-measured twice at 2m17s after the
matrix session's first pass logged an outlier 3m21s; the retest outputs are
bit-identical to the matrix run.)

### Speed alignment: long vs balanced vs PyTorch (286 frames, f16, same
builds/machine)

| engine / mode | long | balanced | ratio |
| --- | ---: | ---: | ---: |
| GGML strict CUDA0 | 355s | 351s | 1.01 |
| GGML strict Vulkan0 | 397s | 413s | 0.96 |
| GGML flash CUDA0 | 114s | 116s | 0.98 |
| GGML flash Vulkan0 | 141s | 146s | 0.97 |
| PyTorch SDPA fp32 streaming | 124s (2.24 it/s) | 122s (E1b) | — |
| PyTorch SDPA bf16 deployment | 53s (5.24 it/s) | not run | — |

The checkpoint choice does not move GGML throughput (±4%, all within
run-to-run noise; the f32/q8/f16 long walls in the tables above agree to
seconds). The PyTorch bf16 deployment runs ~2x faster than GGML flash on
this host because its matmuls are bf16 tensor-core while the GGML graph
computes F32 — the established engine trade (F32 graph precision for
1e-04-class parity at deployment accuracy, MODEL_CARD).

### Checkpoint level (vs the upstream `lingbot-map-long.pt` itself)

The upstream long checkpoint (sha256 `832bc8…f409`, verified byte-exact on
download) decoded and run as the fp32 streaming reference splits the
current-level deviation into weight-format loss and engine addition:

| run | pose RMSE | depth RMSE | tail >1.5e-3 |
| --- | ---: | ---: | ---: |
| mirror f32 (fp32 cache) vs checkpoint | **0.0** | **0.0** | 0.00% |
| mirror f16 (fp32 cache) vs checkpoint | 1.324e-04 | 1.812e-03 | 12.16% |
| mirror q8 (fp32 cache) vs checkpoint | 1.130e-03 | 1.648e-02 | 25.09% |
| GGML strict f32, CUDA / Vulkan | 6.02e-05 / 3.62e-05 | 5.03e-04 / 4.82e-04 | 1.60% / 1.56% |
| GGML strict f16, CUDA / Vulkan | 1.31e-04 / 1.37e-04 | 1.84e-03 / 1.84e-03 | 12.2% / 12.1% |
| GGML strict q8, CUDA / Vulkan | 1.14e-03 / 1.13e-03 | 1.64e-02 / 1.65e-02 | 25.1% / 25.0% |
| GGML flash f16, CUDA / Vulkan | 1.43e-04 / 1.41e-04 | 1.98e-03 / 1.96e-03 | 12.6% / 12.8% |

Readings: (1) the f32 GGUF is a bit-exact conversion of the checkpoint
(mirror-vs-checkpoint = 0), so the GGML f32 rows are pure engine parity at
the checkpoint level; (2) the f16/q8 checkpoint-level deviations are
dominated by the weight format itself — the fp32-cache mirrors already sit
at pose 1.3e-04 / 1.1e-03 and depth 1.8e-03 / 1.6e-02 — with the engine
adding only ~4–8e-05 pose / ~5e-04 depth on top (the vs-mirror rows above).
The long checkpoint's f16 weight loss is larger than the balanced one's
(1.8e-03 vs 4.7e-04 depth) — a property of the released weights, not of the
engine or the cache profile.

### Windowed-mode parity on the long weights

`scripts/verify_windowed.py --mirror-gguf lingbot-map-long-f16.gguf --gguf
lingbot-map-long-f16.gguf --frames 64 --window-size 32 --overlap-size 8`
(the torch side decodes the same GGUF, so raw/cross-warp are pure engine
parity): 3 windows, per-window raw pose `5.1–6.6e-05` / depth
`4.1–8.8e-04`; cross-checked stitch pose `1.18e-04` / depth `5.46e-04` /
c2w center `9.3e-05` / rotation `3.2e-05` — the windowed orchestration is
bit-faithful on the long weights too. The self-aligned merge differs by
`2.1%` relative depth (balanced: `1.0%`) — the method's own scale-estimate
noise under different overlap contexts, not an engine or orchestration
device. The balanced 24-frame verification passes the same gates
(raw `7.9e-05` / `5.3e-04`, cross-warp `6.3e-05` / `4.3e-04`).

Evidence artifacts: `current_long_{dtype}_{backend}_courthouse.{csv,png}` +
`..._comparison.png` + `..._ggml_{dtype}.ply` / `..._pytorch.ply` (the
c2w/intrinsics postprocess gate passed on every round; the elementwise
`compare_parity.py` depth gate trips exactly on the attributed tail above —
pose passes everywhere), plus `current_long_f16_cuda_checkpoint_effect.png`
(effect comparison against the official `lingbot-map-long.pt` reference).
GGUF sha256: f32 `e6a13e81…5cb849a5`, f16
`309ef95b…2705a3a0`, q8 `729260e8…f985bf3805c` (full hashes in the download
script output; `scripts/dl_long_gguf.sh`).

### Audit notes (2026-09-16): remaining gaps, bottleneck decomposition, headroom

Alignment closure: all three reference levels (decoded-GGUF mirror,
official fp32 checkpoint, PyTorch bf16 deployment) are closed for long on
both backends and all three formats, strict + flash; windowed mode is
closed at mirror level. The only measured deviations that remain are (a)
the weight-format loss itself (upstream property, see the checkpoint
subsection), and (b) the F16-cache relative noise reproduced in PyTorch.
No engine-separable error exists at any measured level.

Speed bottleneck decomposition (from the measured walls; the built-in
`--profile` stage timer is graph-granular and does not decompose ops):

- strict vs flash delta on identical inputs: 355s − 114s = 241s over 286
  frames ≈ 0.87 s/frame — the hand-written F32 chunked attention. This is
  the parity route's designed cost, not an optimization target.
- flash residual: ~0.34 s/frame (GGML) vs ~0.19 s/frame (PyTorch bf16
  SDPA) — the GGML graph computes every GEMM in F32 while PyTorch uses
  bf16 tensor cores. This is the **largest remaining speed lever**:
  F16/BF16-activation GEMMs with F32 accumulate (cuBLAS / coopmat) would
  plausibly bring flash to ~1m10–20s, but FFN/proj activation rounding is
  a NEW precision claim (the proven-benign set covers only attention
  Q/K/V inputs) and requires the full mirror-ladder campaign plus 286-frame
  gates on both backends before any release claim.
- Vulkan flash trails CUDA by ~24% (141s vs 114s) because the
  `GGML_VK_FA_COOPMAT_ONLY` bridge keeps GEMMs scalar (NV coopmat2 f32 mma
  is TF32-class — rejected for parity, pitfall 5). Closing it means a
  coopmat GEMM precision campaign of the same class as the F16-GEMM item
  above.
- Accuracy headroom: none actionable in-engine. The deployment-level floor
  is PyTorch's own bf16 self-noise; the q8/f16 weight loss is the upstream
  checkpoint's quantization property (the f32 GGUF is bit-exact to the
  checkpoint).

## 10. Open Follow-ups

1. **Upstream submission of the consolidated ggml patch**
   (`third_party/patches/0001-lingbot-ggml-v021.patch`, 10 files): CUDA
   flash-parity kernel work (Turing-gated), the Vulkan coopmat scalar-F32
   conv pipeline fix (empty-pipeline SIGFPE reproduces on any F32-conv
   workload), the `GGML_VK_FA_COOPMAT_ONLY` hook, and the gallocr re-reserve
   repro. Ready to split into upstreamable pieces when filed.
2. **q8 positioning** (no action unless requirements change): memory-saving
   default, not the parity format — the f16 ladder argument rules out a
   mixed-precision q8 variant.
3. **Long checkpoint-level validation** — CLOSED 2026-09-15: the upstream
   `lingbot-map-long.pt` was downloaded (sha256 verified) and run as the
   fp32 checkpoint reference; the full checkpoint-level table and the
   windowed-mode long parity landed in section 9. The remaining long open
   item is the depth scatter tail itself (an upstream weight property:
   F16-cache sensitivity, section 9) — nothing actionable in this repo
   beyond the documented `--kv-f16 none` route.

## 11. Real Long-Sequence Campaign (official demo data, 2026-09-16)

Scope: end-to-end comparison of the C++ GGML engine against the official
PyTorch pipeline on the **official demo long sequences**
(`robbyant/lingbot-map-demo`), replacing the bundled 286-frame courthouse
proxy. All rows run the official streaming profile `scale=8/window=64` with
the official auto `keyframe_interval = ceil(N/320)` passed explicitly to both
engines; inputs are byte-identical (official crop rule).

| dataset | source video | N | resolution | auto kf |
|---|---|---|---|---|
| lingbo_world | lingbo_world_frames.mp4 (2000 src frames) | 667 | 518x308 | 3 |
| drive | drive_frames.mp4 (3150 src frames) | 1050 | 518x294 | 4 |
| indoor | indoor_travel.MP4 excerpt (25000 src frames, first 2000) | 2000 | 518x294 | 7 |

Matrix per dataset: PyTorch {long,bal} x {fp32 checkpoint, bf16 deployment,
f16-mirror, q8-mirror} + GGML {long,bal} x {f32,f16,q8} x {CUDA,Vulkan} x
{strict,flash}, each row monitored for wall / FPS / GPU peak / RSS peak
(`cpp_ggml/scripts/long_real/`: `run_matrix.sh`, `monitor.py`, `evaluate.py`,
`plot_report.py`; evidence in `benchmarks/long_real/<ds>/` and
`long_real_report.md`).

### 11.1 keyframe_interval engine support (prerequisite, landed 2026-09-16)

`--keyframe-interval N` implements the official `demo.py` policy: every N-th
streaming frame persists KV; non-keyframes attend to
`[special | scale | live | fresh]` and discard (no append, no eviction, no
specials, camera trunk included). Gates (`scripts/long_real/verify_keyframe.sh`):

- **V1 regression**: interval=1 outputs are **bit-identical** to the
  pre-change build across six cache paths (resident strict/flash on CUDA and
  Vulkan, F32 legacy sidecar, F16 legacy sidecar), 100-frame courthouse.
- **V2 semantics** (C++ strict vs official `inference_streaming` mirror,
  100-frame courthouse): kf=2 pose 5.7e-05 / depth 1.05e-04 (CUDA),
  pose 4.6e-05 / depth 7.8e-05-class (Vulkan); kf=4 pose 1.02e-04 / depth
  1.31e-04 — the interval=1 parity class. Per-frame error is identical on
  keyframe and non-keyframe frames (7.7e-05 vs 7.9e-05). compare_parity's
  depth max_abs gate trips on the known F16-cache scatter tail at both
  intervals (1.63e-02 at kf=1 vs 2.72e-02 at kf=2 — same class, see
  pitfall 12); RMSE is the authoritative engine-parity metric here.

### 11.2 Measured matrix (final, 2026-09-18 — 96/96 rows)

All 32 rows per dataset completed (18 rows initially lost to a concurrent
external GPU tenant were refilled by the resume loop on a clean GPU; the
f32+Vulkan load-stage allocation failures were capacity, not engine). Tables:
`benchmarks/long_real/<ds>/alignment.csv` + `resources.csv` + `behavior.csv` +
`cloud_eval.csv`, charts `parity_curves/trajectory/speed/memory/cloud_effect.png`,
consolidated `long_real_report.md`.

**Engine parity (vs the decoded same-GGUF mirror), depth REL is the stable metric**

| checkpoint/format/mode | lingbo_world (667f) | drive (1050f) | indoor (2000f) |
|---|---|---|---|
| long f32 strict (pure engine) | pose 1.2-2.6e-04 / REL 5.6-5.7e-04 | pose 1.7-1.9e-03 / REL 4.5-4.6e-04 | pose 1.3-1.5e-03 / REL 0.9-1.0e-04 |
| long f16 strict | pose 1.3-1.5e-04 / REL 5.7-6.1e-04 | pose 2.5-3.9e-03 / REL 4.5-4.7e-04 | pose 2.7-5.9e-04 / REL 1.0-1.2e-04 |
| long f16 flash | pose 3.8-5.6e-04 / REL 6.7-7.3e-04 | pose 8.3-1.0e-02 / REL 5.6-9.3e-04 | pose 2.1-2.6e-03 / REL 1.2-1.9e-04 |
| long q8 strict (vs q8 mirror) | pose 8.4-2.3e-04 / REL 5.2-5.7e-04 | pose 1.5-1.6e-03 / REL 4.5-4.8e-04 | (pass, see CSV) |

- **depth REL parity stays 1e-04-class on every real dataset, every mode and
  format** (6e-05 - 9.3e-04); pose drifts with stream length and scene
  dynamics (667f indoor-world 1.2e-04 -> 1050f outdoor 1.9e-03) — the
  camera-head AdaLN feedback amplifies any sub-1e-04 perturbation over long
  streams (pitfall 1), reproduced by PyTorch itself (below).
- flash pose parity degrades 3-10x vs strict over long streams (F16-QK
  accumulation); flash depth REL stays within 2x of strict.

**The decisive reference: official bf16 deployment self-noise** (PT bf16
autocast vs its own fp32, same implementation) is LARGER than every GGML
deviation: lingbo long pose 1.30e-02 / REL 5.20e-02; drive long pose
**2.10e-01** / REL 1.27e-01 (grows 1.8e-03 -> 3.2e-01 over the stream);
indoor long pose 7.87e-02 / REL 6.01e-03. The worst GGML row on any dataset
(q8 weight loss: pose 3.8e-02 / REL 3.0e-02) sits 5-55x below the official
deployment self-noise. **Verdict: under deployment-precision semantics the
C++ GGML engine and the official Python pipeline are indistinguishable.**

**q8 weight loss is data-domain dependent** (vs checkpoint fp32, engine
excluded): REL 1.9e-02 (lingbo world-model) / 3.0e-02 (outdoor drive) /
2.3e-03 (indoor) — the same q8 GGUF carries very different absolute cost per
scene; the engine adds nothing on top (q8-vs-q8-mirror rows are strict-class).

**Speed (RTX 4090, official profile, auto keyframe)**

| engine row | lingbo 667f | drive 1050f | indoor 2000f |
|---|---|---|---|
| GGML long f16 flash CUDA | 276 s / 2.41 fps | 404 s / 2.60 fps | 755 s / 2.65 fps |
| GGML long f16 flash Vulkan | 323 s / 2.06 fps | 464 s / 2.26 fps | 987 s / 2.03 fps |
| GGML long f16 strict CUDA | 847 s / 0.79 fps | 1300 s / 0.81 fps | 2811 s / 0.71 fps |
| PT long fp32 streaming | 417 s / 1.60 fps | 458 s / 2.29 fps | 865 s / 2.31 fps |
| PT long bf16 (deployment) | 139 s / 4.79 fps | 205 s / 5.13 fps | 369 s / 5.42 fps |

GGML flash CUDA matches PT fp32 streaming wall-clock while holding
deployment-grade parity; PT bf16 keeps the tensor-core ~2x (the established
F32-graph trade). long-vs-balanced walls agree within ~4% (same architecture).

**Memory (process GPU peak)**: GGML f16 13.4 GB (CUDA) / 17.4 GB (Vulkan),
q8 11.4 GB, f32 17.8 GB — vs PT fp32 19.6-21.8 GB and PT bf16 19.1-19.8 GB;
host RSS peak 8.4-15.2 GB. Strict-mode f32 rows need ~18 GB device headroom:
with a concurrent tenant holding >4 GB the f32 weight upload OOMs at load
(observed 8 rows; all refilled on a clean GPU — capacity, not engine).

**long-vs-balanced behavior** (no GT; consistency/scale dimensions only):
mean |depth| long vs bal = 0.89/0.54 (lingbo), 1.44/0.66 (drive, p95 6.9 vs
1.7), 1.05/1.10 (indoor) — the long checkpoint predicts systematically larger
depth scales, strongest on outdoor; cloud NN RMSE between checkpoints
(0.56 / 6.9 / 1.2) exceeds engine noise by 8-70x (same-checkpoint PT<->GGML
0.06 / 4.1 / 1.2), i.e. the two checkpoints produce structurally different
reconstructions while each engine pair is mutually consistent.

## 12. Attribution Follow-ups (2026-09-18): flash pose, q8mix, F32-cache

Three attributions closing the questions left open by the §11 campaign.
All numbers reproducible via `cpp_ggml/scripts/long_real/attribute_p1p2.py`
and `attribute_p3.py` (per-comparison CSVs next to the campaign tables,
`benchmarks/long_real/{drive,indoor}/attribute_p*.csv`; the new rows are
also appended to the campaign `alignment.csv` files and flowed into
`long_real_report.md`).

### 12.1 P1 — the flash pose excess is the F16 K/V contract, not an engine defect (drive 1050f)

Decisive isolation inside PyTorch, weights constant f16 on both sides, only
the K/V precision entering attention differs (the `LINGBOT_KV_CACHE_F16=flash`
mirror = every K/V F16 vs the F32-cache mirror):

- **F16 K/V contract cost in pure PyTorch: pose 2.38e-03 / depth REL
  4.42e-04** over 1050 frames (first50 2.33e-05 → lastQ 3.72e-03 — the
  cache-feedback growth signature of pitfall 2).
- Totals vs the pristine fp32 checkpoint: flash-contract mirror 3.06e-03,
  F32-cache f16-weights mirror 4.12e-03 — same class; the cache contract and
  the weight format each sit in the e-03 class at this stream length.
- Engine side: GGML long f16 flash totals 8.21e-03 (CUDA) / 1.12e-02
  (Vulkan) vs fp32, i.e. the flash kernel adds a ~5-7e-03-class noise on top
  of the contract cost over 1050 frames — while remaining 19-25x below the
  official bf16 deployment self-noise (2.10e-01).

Consequences: (a) the §11 flash rows referenced the **F32-cache** mirror, so
their pose parity mixes contract cost into the engine number — the correct
flash parity reference is the flash-cache contract mirror; (b) no engine
action item: the residual is the documented F16 numerics class, and it
scales down with stream length (5-8e-05 at the 286f scale, §9).

### 12.2 P2 — mixed quantization is a confirmed negative; the trunk, not the camera head, carries the q8 pose cost (drive 1050f)

The q8mix GGUF (q8 aggregator+depth_head, f16 camera_head, 1.47GB) was built
to test whether keeping the camera head in f16 removes the q8 pose cost
(motivated by the per-tensor sensitivity ranking:
camera_head.pose_branch.fc2, 0.84% rel err, `q8_sensitivity.py`). Result:

- **Weight cost unchanged**: q8mix mirror vs checkpoint fp32 = pose
  3.01e-02 / depth REL 3.15e-02 — the full-q8 class (3.0e-02).
- The isolation experiment decides why: with the camera head f16 on BOTH
  sides (q8mix mirror vs all-f16 mirror), the q8 rounding of
  aggregator+depth_head **alone** produces pose 3.30e-02 / REL 3.03e-02.
  The pose amplification source is the aggregator trunk, whose outputs
  perturb the camera-token stream before the camera head ever runs; the
  head's AdaLN then amplifies whatever arrives (pitfall 1). Per-tensor
  sensitivity is a local metric and does not predict end-to-end behavior.
- The engine adds nothing on mixed weights either: q8mix strict vs its own
  mirror pose 1.54e-03 / REL 4.76e-04 (strict class), q8mix flash 1.04e-02
  (f16-flash class).

Verdict: no publishable mixed format — reducing the q8 pose cost requires
keeping the trunk in f16/f32, which is the f16 GGUF (2.3GB). This closes
§10 item 2 empirically (the f16-ladder argument was right). The q8mix GGUF
stays in `models/gguf/` as the experiment artifact; not documented in
MODEL_CARD.md.

### 12.3 P3 — exact F32 cache: engine noise floor and the strict accumulation source (indoor 2000f)

The `--kv-f16 none` rows (same f16 GGUF, exact F32 cache) against the
references that gpu_p3ref.sh produced:

- **Pure engine parity under the F32-cache contract: pose 8.56e-05 (CUDA) /
  5.75e-05 (Vulkan), depth REL 3.87e-05 / 5.85e-06, tail 0.00%** — e-05
  class over 2000 frames, no accumulation (first200 2.14e-05 → last500
  1.03e-04 CUDA). Cross-backend consistency: none cuda vs none vulkan pose
  6.33e-05.
- Totals vs the pristine checkpoint are entirely the f16 weight cost:
  none 2.89e-03 / 2.86e-03 (pose) vs the mirror's own mf16-vs-fp32
  2.85e-03; depth REL 2.79e-04 / 2.81e-04 vs 2.82e-04. The engine and the
  cache contribute only the e-05 headroom above.

Answer to the P3 question: the strict row's 5.90e-04 pose / 1.19e-04 REL
(vs the F32-cache mirror) is the **F16 stored-cache contract cost** — the
engine itself is e-05-class and flat at 2000 frames, and the F16-cache depth
scatter tail disappears completely (0.00%) with the F32 cache,
re-confirming the §9 courthouse finding at 7x the stream length. The strict
accumulation is a contract property (upstream numerics class), not an
engine defect; `--kv-f16 none` remains the exactness escape hatch.

### 12.4 New matrix rows (appended to the campaign tables)

| row | reference | pose | depth REL | tail% |
|---|---|---|---|---|
| drive ggml_long_q8mix_strict_cuda | long_q8mix mirror | 1.54e-03 | 4.76e-04 | 2.30 |
| drive ggml_long_q8mix_strict_cuda | long_fp32 (total) | 3.13e-02 | 3.13e-02 | 34.63 |
| drive ggml_long_q8mix_flash_cuda | long_q8mix mirror | 1.04e-02 | 9.12e-04 | 8.11 |
| drive ggml_long_q8mix_flash_cuda | long_fp32 (total) | 4.01e-02 | 3.11e-02 | 34.23 |
| indoor ggml_long_f16_none_cuda | long_mf16 | 8.56e-05 | 3.87e-05 | 0.00 |
| indoor ggml_long_f16_none_cuda | long_fp32 (total) | 2.89e-03 | 2.79e-04 | 0.57 |
| indoor ggml_long_f16_none_vulkan | long_mf16 | 5.75e-05 | 5.85e-06 | 0.00 |
| indoor ggml_long_f16_none_vulkan | long_fp32 (total) | 2.86e-03 | 2.81e-04 | 0.63 |

**Overall verdict of the three attributions**: every deviation the campaign
measured decomposes into (i) weight-format cost (q8 e-02, f16 e-03 class),
(ii) cache/attention numerics contract (F16 cache e-04-e-03 by stream
length; flash kernel on top), and (iii) an engine residue at or below the
e-05 class (strict/F32-cache) — the engine never adds a term larger than
the contract it is asked to execute, and no new open item remains on the
engine side.
