# github: https://github.com/Asher-1/lingbot-map-ggml

# LingBot-Map GGUF Models

Download the released GGUF files from [Asher-1/lingbot-map-gguf](https://huggingface.co/Asher-1/lingbot-map-gguf/tree/main) into `cpp_ggml/models/gguf/`. The original PyTorch checkpoints (`lingbot-map.pt`, `lingbot-map-long.pt`) belong in `cpp_ggml/models/pytorch/`.

| Model | Intended use | Verified backends | Accuracy contract | Memory note |
|---|---|---|---|---|
| `lingbot-map-q8.gguf` | Default deployment on 12 GiB GPUs | CUDA and Vulkan 286-frame | Independent decoded-GGUF PyTorch reference, strict `atol=rtol=1e-3`; end-to-end deviation vs the official fp32/bf16 checkpoint `1.31e-03` pose / `3.16e-03` depth RMSE over the full 286-frame stream (`reconstruction_compare_q8_286frames_vs_officialpt_20260910.csv`) — quantization loss accumulates through the streaming cache, so q8 is NOT the full-alignment format (use f16) | Lowest model storage among verified formats |
| `lingbot-map-f16.gguf` | Higher-fidelity deployment; the weight format for full end-to-end alignment with the official PyTorch pipeline | CUDA and Vulkan 286-frame | Same strict contract and streaming cache semantics; end-to-end deviation vs the official fp32/bf16 checkpoint `1.72e-04` pose / `4.72e-04` depth RMSE over the full 286-frame stream (`reconstruction_compare_f16_286frames_vs_officialpt_20260910.csv`) — two orders of magnitude below the official bf16 deployment self-noise | Fits the tested 12 GiB target only with the bounded streaming path |
| `lingbot-map-f16-gct-camera-f32.gguf` | Numerical-diagnosis mixed-precision model | Not a deployment target | Isolates global-attention/Camera F16 rounding; no separate release parity claim | 432 global-attention and 69 Camera tensors are F32; all other tensors are F16 |
| `lingbot-map-f32.gguf` | Reference-grade weights (deepest precision) | CUDA and Vulkan, 286-frame flash mode | End-to-end deviation vs the official fp32/bf16 checkpoint `1.29e-04` pose / `1.22e-04` depth RMSE over the full 286-frame flash stream (CUDA; Vulkan 7.25e-05 / 1.35e-04) — the deepest graph-parity evidence | ~4.6 GB weights; +36% per-frame wall vs q8 (RTX 4090 CUDA, 8-frame incl. load: 6.7 s vs 5.0 s) |
| `lingbot-map-q4.gguf` | Storage experiment | Not release-validated | No parity claim | Does not make the 8-frame 518x518 scale pass fit in 12 GiB |

`lingbot-map-f16-gct-camera-f32.gguf` is not required to run LingBot-Map. It
was made for a bounded A/B test: retain the 24 global-attention blocks and the
CameraCausalHead in F32 while leaving the rest in F16, then determine whether an
observed pose drift comes from those components. Use `lingbot-map-f16.gguf` for
the validated F16 deployment path.

## Long-checkpoint conversion (`lingbot-map-long-*.gguf`)

The three `long` GGUFs convert the upstream
[`lingbot-map-long.pt`](https://huggingface.co/robbyant/lingbot-map/blob/main/lingbot-map-long.pt)
checkpoint (sha256 `832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409`,
4,632,303,465 bytes) with the same `scripts/convert_lingbot.py` defaults as the
balanced models. The long checkpoint is **architecture-identical** to the
balanced one — 1342 tensors, identical names and shapes (verified directly
against both `.pt` files), with 1341 of 1342 weight values differing — so the
C++ graph, GGUF metadata and every engine option apply unchanged; only the
weights differ.

| Model | Intended use | Verified backends | Accuracy contract | Memory note |
|---|---|---|---|---|
| `lingbot-map-long-f16.gguf` | Long-checkpoint deployment | CUDA and Vulkan, 286-frame strict + flash at `scale=8/window=64` (2026-09-15) | Graph parity vs the decoded same-GGUF mirror over the full 286-frame stream: strict pose `3.4–6.9e-05` (backends) / depth `~5e-04` absolute (= REL `2.2e-04`; the absolute tail above 1.5e-03 covers ~1.3–2.1% of pixels **because the long checkpoint predicts a 3.3x larger depth scale on this scene** — the tail reproduces in PyTorch with the same F16 cache, so it is not an engine effect); flash pose `5.1–8.5e-05` / depth `6.4–9.2e-04` (same class). The exact-F32-cache route (`--kv-f16 none`) holds depth REL `2.2e-05` with 0.00% tail | Same size as balanced f16 |
| `lingbot-map-long-q8.gguf` | Memory-bound long-checkpoint deployment | CUDA and Vulkan, 286-frame strict + flash (2026-09-15) | Same graph-parity class as long f16 (strict pose `3.8–6.9e-05` / depth `~4.9e-04`); the depth tail is weight-format independent (cache rounding, see above) | Same size as balanced q8 |
| `lingbot-map-long-f32.gguf` | Reference-grade long weights | CUDA and Vulkan, 286-frame strict + flash (2026-09-15) | Same graph-parity class (strict pose `3.6–6.0e-05` / depth `~5e-04`) — the deepest-precision long weights land on the same cache-precision depth tail as f16/q8, confirming the attribution | Same size as balanced f32 |

Walls (286 frames, RTX 4090, native aspect 518x294, `scale=8/window=64`):
strict CUDA 5m54–5m56s / Vulkan 6m35–6m38s; flash CUDA 1m53–1m55s /
Vulkan 2m21–2m28s (f32 2m17s; re-measured twice after a 3m21s matrix-session
outlier). Evidence:
`cpp_ggml/benchmarks/validation_report.md` section 9,
`cpp_ggml/benchmarks/current_long_*`.

Validation status (2026-09-15/16, checkpoint level CLOSED): the long GGUFs
carry the full mirror-level 286-frame contract at the shipped streaming
profile (pose at the balanced class; depth REL 2.2e-04 with the absolute
scatter tail explained by the long checkpoint's 3.3x larger predicted depth
scale — the tail reproduces in PyTorch under the same F16 cache, so it is
not an engine effect; 0.00% tail on the exact F32 cache) AND the
checkpoint-level rows against the upstream `lingbot-map-long.pt` (sha256
verified): the f32 GGUF is a bit-exact conversion of the checkpoint, so its
GGML rows are pure engine parity (pose 3.6–6.0e-05); the f16/q8
checkpoint-level deviations are dominated by the weight format itself
(mirror-vs-checkpoint pose 1.3e-04 / 1.1e-03, depth 1.8e-03 / 1.6e-02) with
the engine adding only ~4–8e-05 pose / ~5e-04 depth. Deployment view: all
twelve GGML long rows sit at PyTorch's own bf16 self-noise versus the
bf16 SDPA deployment (pose 2.06–2.28e-03 / depth REL 3.95–4.04e-02 vs
self-noise 2.08e-03 / 3.95e-02) — no engine-separable error. Speed: GGML
long matches balanced within ±4% in every mode/backend; PyTorch long runs
124s streaming fp32 / 53s bf16 (SDPA) on the same host. Windowed mode
(`--mode windowed`) is verified on the long weights at mirror level
(per-window raw pose 5.1–6.6e-05, cross-checked stitch 1.2e-04 / 5.5e-04 —
`scripts/verify_windowed.py`). The elementwise `compare_parity.py` depth
gate trips on the depth-scale-weighted absolute tail (pose passes
everywhere); treat the RECONSTRUCTION RMSE gates plus the `--kv-f16 none`
route as the authoritative long gates. `run_gui.sh` falls back to a long
GGUF only when no balanced GGUF is present; pass `--gguf` explicitly to
force one.

### Experimental format: mixed quantization (`lingbot-map-long-q8mix.gguf`) — closed negative

A 1.47 GB experiment (q8 aggregator+depth_head, f16 camera_head) testing
whether keeping the camera head in f16 removes the q8 pose cost. It does
not: the weight cost vs the fp32 checkpoint (pose 3.01e-02 / depth REL
3.15e-02 over the 1050-frame drive stream) stays in the full-q8 class,
because the q8 rounding of the aggregator trunk alone reproduces it (pose
3.30e-02, measured with the camera head f16 on BOTH sides). The pose
amplification source is the trunk feeding the camera tokens, not the
camera-head weights; per-tensor sensitivity rankings (the 0.84% fc2
tensor) do not predict this. Reducing the q8 pose cost requires an
f16/f32 trunk, which is the f16 GGUF. The engine adds nothing on mixed
weights either (strict parity 1.54e-03, the strict class). Attribution:
`validation_report.md` §12.2 and
`benchmarks/long_real/drive/attribute_p1p2.csv`. Not a release format;
kept only as the experiment artifact.

### End-to-end alignment with the official PyTorch pipeline (2026-09-10)

The engine is weight-format agnostic: run against the same q8 weights, GGML
(pose `2.19e-04` / depth `1.56e-03` vs the official checkpoint, 8 frames at
the native aspect) and PyTorch decoding the same q8 GGUF (`2.09e-04` /
`1.56e-03`) deviate from the official checkpoint by the same amount — the
entire gap is the weight format, not the engine. With the f16 GGUF the
deviation drops to `1.72e-04` pose / `4.72e-04` depth over the full
286-frame stream. Preprocessing is bit-identical between the GGML GUI and
the official loader (`max_abs_diff=0` on courthouse). Full matrix:
`benchmarks/validation_report.md`, section "End-to-End Alignment Against
the Official PyTorch Pipeline". Choose the f16 GGUF when the reconstruction
must match `demo.py` as closely as possible over long streams; the f32 GGUF
is the deepest-precision reference; choose q8 when memory is the binding
constraint.
Per-frame inference wall (RTX 4090 CUDA, 8 frames incl. load):
q8 5.0 s / f16 5.6 s / f32 6.7 s. Full-stream wall (286 frames, RTX 4090,
f16 GGUF): strict mode 6m47s (CUDA) / 7m05s (Vulkan) — the device-resident KV
cache keeps the GPU at 100% utilization; flash mode **2m11s** (CUDA) /
**2m39s** (Vulkan) at the same 1e-4-class parity, matching the PyTorch
FlashInfer reference (2m14s). Full matrix:
`benchmarks/validation_report.md`.

`run_gui.sh` (GGML engine) reflects this ranking in its defaults: the f16
GGUF is picked automatically when present (override with `--gguf <path>` or
`GGML_MODEL`), so the out-of-the-box GUI reconstruction is the full-alignment
one; q8 remains downloadable/usable for memory-bound machines. The backend
defaults to CUDA0 when a build-cuda exists (the accuracy/speed reference),
else Vulkan0 (`--backend` to override, auto-selecting/making build-<backend>).

The 8-frame `scale=8/window=64` profile reserves the attention, KV-cache, and
activation graph before execution. At 518x518 that reservation is
28,524,031,984 bytes and fails before inference on the 12 GiB test GPU;
reducing q8's on-disk weights to q4's 671 MiB cannot change the graph size.
Reduced resolutions are supported through the native DINO positional
interpolation (verified against the CUDA reference to `max_abs=1.9e-07`;
see `AGENTS.md`):
392x392 validates the 8-frame profile for q8 CUDA/Vulkan and f16 Vulkan on a
24 GiB card, while 448x448 already reserves 23.1 GB on Vulkan and fails
there. Use the release `scale=1/window=4` profile for 518x518 on 12 GiB
hardware, or the native-aspect 518x294 profile (the identity output of the
official crop rule for the bundled example scenes) for the best quality.

### Keyframe interval (official long-stream policy)

`--keyframe-interval N` implements `demo.py --keyframe_interval`: every N-th
streaming frame persists its KV; the rest attend and discard. The official
demo auto-selects `ceil(N/320)` for streaming runs above 320 frames — pass
the same value on both engines for comparable runs. With the interval set,
the resident special-token segment is sized for stored keyframes rather than
the raw stream length (`--kv-total` stays the total stream length). Verified
against the official `inference_streaming` mirror at interval 2/4
(pose/depth RMSE 1e-04-class, `scripts/long_real/verify_keyframe.sh`) and
bit-identical at interval=1. Real official-demo long-sequence evidence:
`benchmarks/long_real/`.

The deployment contract is not the filename alone. Run `bash cpp_ggml/scripts/run_e2e.sh cuda q8` or `bash cpp_ggml/scripts/run_e2e.sh vulkan f16`; the command builds the selected backend, runs official `courthouse` at the native aspect, compares against an independently decoded matching GGUF in PyTorch with cuDNN disabled, and writes scene/PLY/image evidence under `cpp_ggml/benchmarks/`. The shipped cache profile is `scale=8/window=64` with the persistent F16 KV cache (the same profile as the official PyTorch pipeline), validated end to end on both backends at the native aspect and at 392x392 / 518x378; the bounded `scale=1/window=4` profile remains available for 12 GiB hardware.

### F16 KV cache (LINGBOT_KV_CACHE_F16)

Both GGUFs support the persistent F16 KV cache, which halves the per-frame
streaming cache upload:

- `LINGBOT_KV_CACHE_F16=1` / `--kv-f16 strict` (parity mode, default): cache
  persisted as device-resident F16 payloads, attention stays on the
  hand-written F32 path in exact query chunks. Validated over 286 frames at
  the native aspect 518x294 (pose RMSE 6.1e-05 CUDA / 3.9e-05 Vulkan),
  518x518 `scale=1/window=4` (pose RMSE 1.1e-04), 392x392 (6.9e-04–9.2e-04)
  and the official working point 518x378 `scale=8/window=64`
  (1.0e-04–1.7e-04) on CUDA and Vulkan against a cache-precision-aware
  PyTorch reference; steady state 14.3 GiB at 518x378 (20.2 GiB during the
  scale-pass graph).
- `LINGBOT_KV_CACHE_F16=flash` (fast mode): streaming frames run
  `ggml_flash_attn_ext` (14.3 GiB at 518x378) with F32-effective PV numerics
  (F32 VKQ accumulation + P hi/lo split on CUDA; on Vulkan the coopmat1 FA
  with the P hi/lo split and an F32 output chain, GEMMs kept scalar) and an
  F32 cache for the camera trunk. Validated over the full 286-frame
  courthouse stream at the native aspect 518x294 on BOTH backends and for
  the f16/q8/f32 GGUFs: f16 CUDA pose 1.393e-04 / depth 4.55e-04 at 2m11s
  (FlashInfer-class speed); f16 Vulkan 7.27e-05 / 1.06e-04 at 2m39s
  (vs the official fp32 checkpoint 1.84e-04 / 4.50e-04 — 1.69x faster than
  the scalar FA, 1.21x behind CUDA); f32 CUDA 1.29e-04 / 1.22e-04;
  q8 CUDA/Vulkan 1.33e-03 / 3.17e-03 (the documented q8 weight cost,
  identical to the strict q8 rows).
- The scale pass always keeps exact F32 attention in every mode.

### Exact F32 KV cache (`--kv-f16 none`)

The tightest-parity route: the KV cache stays F32 (legacy sidecar path),
so the F16-cache rounding contract and its depth scatter tail disappear
entirely. Validated on the official long streams at the shipped profile
(`scale=8/window=64`, auto keyframe interval):

| scene | pose parity (vs same-weights mirror) | depth REL | tail > 1.5e-03 |
|---|---|---|---|
| indoor 2000f CUDA | 8.56e-05 (strict F16 cache: 5.90e-04) | 3.87e-05 | 0.00% |
| indoor 2000f Vulkan | 5.75e-05 (strict: 2.67e-04) | 5.85e-06 | 0.00% |
| courthouse 286f | depth REL 2.2e-05 (validation_report.md §9) | — | 0.00% |

Trade-offs: the resident KV payload roughly doubles (indoor 2000f CUDA
process peak 17.4 GB vs 13.4 GB) and the wall roughly doubles (5340 s vs
2811 s — the F32-cache path is the legacy sidecar, not the fused
device-resident one). Attribution (validation_report.md §12.3): the
F16-cache cost is a numerics contract, not an engine defect; `none` is
the exactness escape hatch. GUI: `run_gui.sh --engine ggml --exact` or
`ggml_demo.py --kv_f16 none`.

Non-square resolutions additionally depend on the fixed native DINO
positional resampler (verified to `max_abs=1.9e-07` on the 37x27 grid;
see `AGENTS.md`).

## Measured Full CUDA Reconstruction

On the 286-frame official `courthouse` sequence at the native aspect
(518x294) with the upstream `scale=8/window=64` profile and the persistent
F16 KV cache, q8 CUDA reached pose/depth RMSE `6.14e-05 / 9.75e-05`; Vulkan
q8 reached `3.89e-05 / 9.71e-05` at the same profile. The two backends also
agree directly with each other (pose RMSE 4.26e-05, depth RMSE 2.78e-05).
These figures compare C++ to a separately decoded, same-format GGUF PyTorch
reference, so they measure graph/backend parity rather than checkpoint
quantization loss.

## Companion model: native sky segmentation (`lingbot-map-skyseg-*.gguf`)

`lingbot-map-skyseg-{f32,f16,q8_0}.gguf` are the official `skyseg.onnx`
sky-segmentation network converted for the native ggml runtime
(`scripts/convert_skyseg.py` -> `src/skyseg.cpp`). They let the C++ engine
run the official `--mask_sky` pipeline (sky-pixel confidence zeroing) with
no onnxruntime and no Python in the inference path.

| Variant | Size | Keep-rule agreement (8 frames, official chain) | Latency (RTX 4090, ms/frame) | Notes |
| --- | --- | ---: | ---: | --- |
| `lingbot-map-skyseg-f32.gguf` | 176 MB | 100.00% | 16.4 ms (CUDA0), 25.9 (Vulkan0) | reference precision |
| `lingbot-map-skyseg-f16.gguf` | 88 MB | 100.00% | 17.6 ms (CUDA0), 27.2 (Vulkan0) | **recommended** (default for `--mask_sky`) |
| `lingbot-map-skyseg-q8_0.gguf` | 47 MB | 99.99% (keep-IoU >= 0.9998) | 18.1 ms (CUDA0), 27.9 (Vulkan0) | all backends; on Vulkan the q8_0 kernels dequantize to F16 at load (v0.21 capability chain — `benchmarks/skyseg_ggml.md`) |

Input: 320x320 RGB, ImageNet-normalized (preprocessing is built into the
CLI integration, with the cv2 u8 bilinear resize replicated bit-exactly).
Keep-rule agreement with the official path over the 8-frame courthouse
stream is 100% (f32/f16) and 99.99% (q8_0, weight-quantization noise) on
every backend; tables:
`benchmarks/skyseg_ggml.md`.

The mask postprocess in the CLI replicates the official
`lingbot_map/vis/sky_segmentation.py` chain exactly (min-max normalize, u8
truncate, bilinear resize, keep "resized u8 == 0"), with the OpenCV u8
resize replicated bit-exactly. `lingbot-map-cli ... <skyseg.gguf>
<mask-cache-dir> <visualization-dir>` writes/reads the official
`<folder>_sky_masks`-style PNG cache and official-style
original|mask|overlay panels; `ggml_demo.py --mask_sky` wires all of it and
`run_gui.sh` defaults the cache dirs. skyseg is a fixed-class sky/non-sky
network — the official pipeline has no open-vocabulary segmentation, so
official-demo parity is the complete target.
