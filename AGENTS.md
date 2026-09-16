# AGENTS.md — LingBot-Map GGML Engineering Guide

Engineering knowledge for AI agents and developers working in this repository:
every landed optimization (what problem it solved, where the code lives, how it
was verified), the long-stream reconstruction support, the validation protocol,
the ggml patch management rules, and the hard-won pitfalls. Final measured
numbers and evidence charts: `cpp_ggml/benchmarks/validation_report.md`.

## Project Map

```
cpp_ggml/                  native C++ ggml runtime (the optimization surface)
  src/model.cpp            GCT streaming graph: 24 global blocks + camera head + DPT,
                           KV-cache modes, device-resident cache, options plumbing
  src/backend.cpp/.h       backend init; drives ggml's env-only toggles from options
  src/gguf_loader.cpp      GGUF reader (f32/f16/q8_0)
  include/lingbot_map.h    public API: model, output, model_options, kv_f16_mode
  tools/lingbot-map-cli.cpp  CLI: positional contract + named option flags
  tools/fattn_stream_test.cpp  kernel-level FA-vs-F32 A/B harness
  third_party/ggml/        pinned ggml v0.21.0 submodule (all changes via patch, see below)
  third_party/patches/0001-lingbot-ggml-v021.patch  consolidated patch (10 files)
  models/gguf/             lingbot-map-{q8,f16,f32}.gguf + lingbot-map-long-{q8,f16,f32}.gguf
                           + lingbot-map-skyseg-*.gguf
  models/MODEL_CARD.md     formats, contracts, capacity limits
  benchmarks/validation_report.md  final measured matrix + evidence charts
  scripts/                 conversion, e2e gate, references, plotting
lingbot_map/               official PyTorch model (mirror semantics live here:
                           layers/attention.py, heads/camera_head.py)
ggml_demo.py / run_gui.sh  one-command GUI over either engine
benchmark/                 BSS scene-level evaluation harness
example/courthouse         286-frame validation scene (native 518x294)
```

## Build & Run

```bash
# CUDA backend (accuracy/speed reference)
cmake -S cpp_ggml -B cpp_ggml/build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_ggml/build-cuda --parallel 6
# Vulkan backend
cmake -S cpp_ggml -B cpp_ggml/build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_ggml/build-vulkan --parallel 6

# CLI (positional: MODEL IMAGE.bin BACKEND H W OUT FRAMES; named flags on top)
cpp_ggml/build-cuda/lingbot-map-cli cpp_ggml/models/gguf/lingbot-map-f16.gguf \
    frames.bin CUDA0 294 518 out.lbo 286 \
    --kv-f16 flash --kv-scale 8 --kv-window 64 --scale-frames 8 --kv-total 286 \
    --keyframe-interval 3

# End-to-end parity gate (builds, runs 286 frames, compares vs decoded GGUF)
bash cpp_ggml/scripts/run_e2e.sh cuda q8      # or: vulkan f16

# PyTorch references (286-frame mirror / official checkpoint)
cpp_ggml/scripts/run_pytorch_reference.py MODEL.gguf frames.npy out.npz \
    --device cuda --streaming --scale-frames 8 --kv-cache-scale 8 --kv-cache-window 64 [--gguf MODEL.gguf]

# One-command GUI (either engine)
bash run_gui.sh --engine ggml
```

Inputs: raw binary `float32[N,3,H,W]` in 0..1, preprocessed with the official
crop rule (`load_and_preprocess_images(mode="crop", image_size=518)` — for
`example/courthouse` that is the native 518x294, no resize). Reference `.npy`
inputs are `[1,N,3,H,W]`.

## Mode Contract (kv_f16_mode)

| mode | attention | numerics class | use |
| --- | --- | --- | --- |
| `none` | hand-written F32, F32 cache | exact; heaviest | debugging |
| `strict` (default) | hand-written F32 in exact query chunks; F16 cache upload; device-resident payloads | 1e-4-class parity | bit-level validation route |
| `flash` | `ggml_flash_attn_ext` on F16 K/V with F32-effective PV numerics (below), GEMMs scalar | 1e-4-class parity at FlashInfer speed | deployment speed |

Both parity modes hold ~1e-4 vs the official fp32 checkpoint over the full
286-frame stream; the scale pass always keeps exact F32 attention. Camera
trunk (`camera_head.trunk.*`) always keeps an F32 cache (see pitfalls).

## Landed Optimizations

Each item: problem → change (file pointers) → verification → result. All ggml
changes live in the consolidated patch; all host changes in `cpp_ggml/src`.

### 1. Query-chunked strict attention + persistent F16 KV cache
- Problem: the full `[S_k, S_q, heads]` score tensor is 4.7 GiB/layer at the
  working point (286 frames x 1005 tokens) — impossible to allocate; F32 cache
  inputs alone were 14.2 GB/frame.
- Change: `model.cpp` builds attention in exact query chunks (each chunk
  softmaxes over the full key axis — math unchanged; views materialized with
  `ggml_cont` because Vulkan misreads strided-b `mul_mat`), fused
  `ggml_soft_max_ext_inplace` per chunk (bit-exact vs scale+softmax, verified
  by `tools/softmax_scale_test.cpp`); cache uploaded as F16, cast back per
  layer.
- Result: all profiles/backends at graph parity (6.1e-05 pose q8 CUDA native).

### 2. Device-resident KV cache (long-stream support, 4.4x)
- Problem: the host sidecar did full F32→F16 conversion + PCIe upload +
  reassembly + capture readback per frame (~50 min per 286-frame stream, GPU
  util 15%).
- Change: fixed-capacity device payloads for the global layers — F16
  `[scale | live]` segments + F32 special-token segment, persisted through
  in-graph cast+cpy nodes; the live segment compacts via a gallocr temporary so
  no in-graph copy overlaps itself; the special-token copy and compaction are
  ordered *before* the frame's attention readers, the fresh append *after*
  (the ordering split an earlier prototype missed and regressed pose to 0.9).
- Verification: CUDA/Vulkan 12/100/286-frame A/B **bit-identical** to the
  legacy path (286 covers all 214 evictions).
- Result: **6m47s** (CUDA strict), GPU util 100%, upload ~75 ms (trunk only).
- Current default. `LINGBOT_KV_TOTAL_FRAMES` (CLI `--kv-total`) sizes the
  special segment.

### 3. CUDA flash parity: fattn-mma F32 VKQ + P hi/lo + np==1 direct write
- Problem: `ggml_flash_attn_ext` measured 1.94e-02 pose over 286 frames.
  Attribution (see "Validation protocol" for the harnesses) isolated the
  fattn-mma half2 VKQ accumulator + half-precision rescale + single-F16 P
  conversion as the only parity blockers (F16 K/V and Q→F16 were proven
  benign); upstream master still uses half2 VKQ, so no upgrade fixes it.
- Change (patch, `fattn-mma-f16.cuh`, Turing-gated):
  `T_C_VKQ` half2 → `tile<16,16,float>` (`mma.f32.f16.f16.f32`, same C-fragment
  as the KQ side), float running-max rescale, wide/narrow combine meta and VKQ
  array sizing adjusted to the unpacked layout; P split into F16 hi + F16 lo
  with a second accumulating mma (~22-bit effective weights); np==1 combine
  writes registers → F32 dst directly (skips the half-valued shared tile);
  `model.cpp` keeps the camera trunk on an F32 cache in flash mode.
- Verification: kernel harness rmse 6.7e-05 → 2.6e-06 (≈ input-rounding
  floor); 286-frame pose 1.94e-02 → **7.70e-05**, depth 1.15e-04, wall
  **2m11s** (no tensor-core throughput given up).
- Ladder of rejected alternatives: 3xTF32 simulation (RTX 4090: TF32 peak ≈
  fp32 CUDA-core peak — simulation is slower than SIMT); ggml upgrade (upstream
  VKQ still half2).

### 4. Vulkan flash parity + speed: coopmat1 FA + P hi/lo + F32 output chain
- Problem: Vulkan FA ran a scalar F32 path (exact but 4m29s). Enabling
  cooperative matrices exposed three blockers: coopmat GEMMs round activations
  to F16 (rejected for parity), NV coopmat2's f32 mma is TF32-class (rejected —
  P keeps only 10 mantissa bits, same class as F16), and coopmat1's single-F16
  softmax-weight conversion amplified to pose 2.5e-03 over 286 frames.
- Change (patch, shaders gated on a new `FA_F32ACC` define injected by
  `vulkan-shaders-gen` into the F32-accumulator FA variants only):
  `flash_attn_cm1.comp` splits P into F16 hi/lo staged in a second shared array
  with a second `coopMatMulAdd` into the same F32 accumulator, rowsum
  accumulates raw F32 exponentials; `flash_attn_base.glsl` sets
  `O_TYPE = float` for FA_F32ACC (PV accumulator coopmat, pvsh staging, rescale
  and `tmpshv4` reduction staging were all F16 — the CUDA half2-VKQ error
  class); scalar `flash_attn.comp`'s `Of` switched from `FLOAT_TYPE` to
  `O_TYPE`; shared-memory ledgers updated (`Psh_lo`, vec4-sized `pvsh`/
  `tmpshv4`). Host: `backend.cpp` flash branch sets
  `GGML_VK_FA_COOPMAT_ONLY=1` + `GGML_VK_DISABLE_COOPMAT2=1` (GEMMs stay
  scalar); `GGML_VK_FA_COOPMAT_ONLY` clears the GEMM-side coopmat/fp16 flags
  while `coopmat1_fa_support` is already fixed.
- Verification: 12-frame probe 7.1e-06 pose vs the mirror (no bit-identity
  illusion); 286-frame **2m39s**, pose 7.27e-05 vs the mirror; strict mode
  improved too (f32acc scalar FA picked up the F32 output chain:
  5.49e-05 → 3.38e-06 pose). Full f16/f32/q8 matrix lands 2m30–2m39s.
- Along the way, two upstream-worthy defects fixed: the coopmat2/cm1 conv2d +
  conv_transpose_2d branches only created `_f16_f32` variants, so F32 DPT convs
  looked up an empty pipeline and died with SIGFPE (scalar F32 conv pipelines
  now created there as well); the gallocr stale-memory repro was documented
  (`tools/mul_mat_strided_b_test.cpp` header).

### 5. Explicit options (no environment switches)
- All former `LINGBOT_*` env reads replaced by `lingbot::model_options`
  (`include/lingbot_map.h`): `model::load(path, backend, threads, options)`,
  `backend::init(name, threads, vulkan_fast, vulkan_integer_dot,
  vulkan_fa_coopmat)`, CLI named flags (`--kv-f16|--kv-scale|--kv-window|
  --scale-frames|--kv-total|--vulkan-fast|--vulkan-integer-dot|
  --force-f32-weights|debug/dump flags`), `ggml_demo.py` passes flags.
- The `GGML_VK_DISABLE_*` setenv calls remain only as the documented bridge to
  ggml v0.21's env-only configuration surface, driven by the options.
  `SKYSEG_*` hooks in `skyseg.cpp` are diagnostics-only and stay env-driven.
- Verified bit-identical on rebuild (flash/strict, both backends). The
  ggml env bridge assigns **idempotent target states** on every `init`
  (each `GGML_VK_*` toggle explicitly set *or* unset per call), so repeated
  in-process init calls (mode or model switches) cannot inherit a previous
  call's environment — verified by pre-seeding stale residues and checking
  the outputs stay bit-identical to clean-init baselines in both
  strict→flash and flash→strict orders.
- q8 integer-dot probe: dp4a GEMM variants are exact (bit-identical) but show
  no wall benefit for this model's shapes — stays opt-in (`--vulkan-integer-dot`).

## Long-Stream Reconstruction Support (what makes 286+ frames work)

- **Streaming profile**: `scale=8/window=64` (upstream default) — an 8-frame
  one-shot scale pass followed by per-frame streaming with a 64-frame live
  window. The scale pass is bidirectional and stays exact-F32 in every mode
  (rounding its K/V costs ~1e-2 regardless of kernel).
- **Device-resident cache**: fixed-capacity F16 payloads (scale + live
  segments) per global layer + F32 special-token segment; `--kv-total` sizes
  the special segment to the total stream length; eviction compacts the live
  segment in-graph. The camera trunk keeps a separate tiny F32 cache.
- **Query chunking**: the strict path's attention memory is independent of S_k
  (chunk rule `q_chunk = max(64, min(512MB/(S_k*heads*4), tokens))`).
- **Capacity floors**: the 8-frame scale-pass graph reservation is the hard
  limit (518x518 = 28.5 GB Vulkan; 448x448 fails 24 GiB on Vulkan; 518x378
  CUDA 11.5 GB; native 518x294 fits everywhere). Non-square grids use the
  native DINO positional interpolation (verified vs CUDA reference,
  max_abs 1.9e-07). Peak VRAM: strict 20.2 GiB scale-pass → 14.3 GiB steady;
  flash 14.3 GiB.
- **Verified stream length**: 286 frames (the bundled courthouse scene), the
  entire matrix above; the architecture carries no stream-length term beyond
  the special-token segment sizing.
- **Long-checkpoint GGUFs** (`lingbot-map-long-{f32,f16,q8}.gguf`, converted
  from upstream `lingbot-map-long.pt`, sha256 `832bc8...f409`): the long
  checkpoint is architecture-identical to the balanced one (1342 tensors,
  identical names/shapes), so the graph and every option apply unchanged.
  Full 286-frame `scale=8/window=64` mirror-level validation landed
  2026-09-15 (`scripts/run_long_matrix.sh` + `scripts/long_followup.sh`,
  evidence in `benchmarks/validation_report.md` section 9): strict pose
  `3.4–6.9e-05` on both backends and all three formats (balanced class),
  strict wall CUDA 5m55s / Vulkan 6m36s; flash pose `5.1–8.5e-05`, wall CUDA
  1m53–1m55s / Vulkan 2m21–2m28s. Depth carries a weight-format-independent
  F16-cache scatter tail (~1.3–2.1% of pixels above 1.5e-03, RMSE ~5e-04)
  that is **reproduced in PyTorch itself, not an engine defect**
  (round-2 attribution 2026-09-16: with `LINGBOT_KV_CACHE_F16=1` the mirror
  lands depth REL `2.44e-04` / tail `1.91%` on long vs C++ `2.22e-04` /
  `1.93%`; balanced `1.58e-04` / `0.05%` vs C++ `1.20e-04` / `0.03%`; the
  long:balanced relative-perturbation ratio is only 1.5x and the cache |V|
  scales agree to 1.11x). The dominant amplifier of the absolute metrics is
  the **long checkpoint's 3.3x larger predicted depth scale** on the scene
  (mean |depth| 2.27 vs 0.70; tail pixels are far-range, median |depth| 7.0,
  per-bin relative error ~1e-4 everywhere) — absolute gates are
  depth-scale-weighted, so gate long depth on the relative view. With the
  exact F32 cache (`--kv-f16 none`) the same weights land depth REL
  `2.2e-05` with 0.00% tail. The elementwise depth gate therefore trips for
  long under F16 cache; the authoritative long gates are the RECONSTRUCTION
  RMSE gates and the F32-cache
  route. Checkpoint level CLOSED 2026-09-15: `lingbot-map-long.pt` downloaded
  (sha256 verified) and run as the fp32 checkpoint reference — the f32 GGUF is
  a bit-exact conversion of the checkpoint, so its GGML rows are pure engine
  parity (pose 3.6–6.0e-05); the f16/q8 checkpoint-level deviations are
  dominated by the weight format itself (mirror-vs-checkpoint depth 1.8e-03 /
  1.6e-02) with the engine adding only ~4–8e-05 pose / ~5e-04 depth
  (`validation_report.md` section 9, checkpoint subsection). Windowed mode is
  verified on the long weights at mirror level (per-window raw pose
  `5.1–6.6e-05`, cross-checked stitch `1.2e-04` / `5.5e-04`). The only long
  open item is the depth scatter tail itself — an upstream weight property,
  nothing actionable here beyond the `--kv-f16 none` route. `run_gui.sh`
  falls back to a long GGUF only when no balanced GGUF is present.
- **Windowed mode (offline batch) is an orchestration layer, not engine
  work**: `ggml_demo.py --mode windowed` (`run_gui.sh` passes the flags
  through) splits the sequence per `inference_windowed`'s fixed-interval
  rules, runs every window through the validated streaming primitive in a
  fresh CLI process (= fresh KV cache, keyframe_interval=1), then aligns
  consecutive windows with a numpy port of `_pairwise_alignment` /
  `_warp_predictions` / `_stitch_windows`. pose_enc is **w2c** (decoder:
  `w2c = [quat_mat | pe[:3]]`, c2w sidecar = inverse) — warp the c2w sidecar
  as `R' = R_c2w @ R.T`, `t' = s·t_c2w − R_c2w @ (R.T t)` to stay the exact
  decoder image of the official warp. Verified against the official PyTorch
  `inference_windowed` by `scripts/verify_windowed.py`: per-window raw pose
  `7.9e-05` / depth `5.3e-04`; cross-checked stitch (same alignment inputs)
  `6.3e-05` / `4.3e-04` — orchestration is bit-faithful; the self-aligned
  merged outputs differ ~1e-2 because each side's depth-ratio scale estimate
  carries the method's own noise floor.
- **Keyframe interval (official long-stream policy, engine-supported)**:
  `--keyframe-interval N` (`model_options::keyframe_interval`) implements
  `demo.py --keyframe_interval`: every N-th streaming frame persists its KV
  (`is_keyframe = (i - scale) % N == 0`); non-keyframes attend to
  `[special | scale | live | fresh]` and discard — no append, no eviction,
  no specials (mirror `_set_skip_append`; C++ `graph_builder::skip_append`
  guards the resident pre-evict/append, both capture pushes, and the camera
  trunk). demo.py auto-selects `ceil(N/320)` for streaming above 320 frames —
  pass it explicitly on both sides for deterministic comparisons; the
  resident special segment is sized for stored keyframes, not raw stream
  length. Verified (`cpp_ggml/scripts/long_real/verify_keyframe.sh`):
  interval=1 is bit-identical to the pre-change build on all six cache
  paths; interval 2/4 land pose/depth RMSE 5.7e-05/1.05e-04 and
  1.02e-04/1.31e-04 vs the official mirror (the 1e-4 class of interval=1).

## Validation Protocol (gate hierarchy)

1. **Kernel level** — `tools/fattn_stream_test.cpp` (FA vs hand-written F32 on
   identical inputs, S_k/S_q sweep), `tools/softmax_scale_test.cpp`,
   `tools/chunk_attn_bench.cpp`, `tools/mul_mat_strided_b_test.cpp`
   (gallocr repro). Use for any attention-kernel change before model runs.
2. **Mirror level** (graph parity) — decoded-GGUF cache-precision-aware
   PyTorch reference: isolates graph/backend parity from weight loss. 12-frame
   A/B for fast iteration; per-frame pose/depth RMSE + allclose violations.
3. **Checkpoint level** (end-to-end) — official fp32 `.pt` reference over the
   full 286-frame stream; also the wall-clock gate (flash must stay ≤ ~2m40s,
   strict ≤ ~7m).
4. **Bit-identical A/B** — every cache/layout refactor must reproduce the
   predecessor's bytes on 12/100/286-frame runs (286 covers all evictions)
   before any precision work is layered on.
5. **Cross-format smoke** — q8/f32 rows must equal the documented weight cost
   only (q8: 1.31e-03/3.16e-03 vs official fp32; anything above means the
   route adds noise).
6. **Postprocess gate** — `compare_postprocess.py` on the c2w/intrinsics
   sidecars every run.
7. **GUI-chain gate** — `scripts/verify_gui_chain.py` runs the exact
   `run_gui.sh` code path (`ggml_demo.load_official_frames` →
   `run_ggml_inference` → `build_pred_dict`) and compares localization
   (pose), trajectory (camera centers via the official pose decoder),
   mapping (depth), reconstruction (official unproject, conf>1.5 viewer
   rule) and intrinsics against the official fp32 streaming reference and
   the cache-aware mirror, for CUDA/Vulkan x strict/flash. Note: two
   engines cannot share one GPU concurrently (the PyTorch session's ~16 GB
   collides with the scale-pass graph reservation) — run them sequentially.

Diagnostic technique worth reusing: a 12-frame strict-vs-flash native A/B
where depth is bit-identical but pose diverges means the camera/special-token
path; PyTorch mirror A/Bs (flag on/off) are cheap and decisive for
attribution.

## Patch Management (third_party/ggml)

- The submodule is pinned at ggml v0.21.0 and **must always satisfy**:
  `worktree == HEAD + third_party/patches/0001-lingbot-ggml-v021.patch`.
  After any ggml edit: regenerate the patch, then verify the round trip —
  `git apply --reverse --check`, `apply -R`, `apply`, `git diff` matches the
  regenerated patch. CMake replays the patch on configure.
- The patch currently carries: CUDA flash-parity kernel work (Turing-gated),
  the Vulkan `FA_F32ACC` shader work, the coopmat scalar-F32 conv pipeline fix,
  the `GGML_VK_FA_COOPMAT_ONLY` hook, FA layout doc wording. Upstream
  submission (split into pieces) is the main open follow-up.

## Known Pitfalls (hard-won; do not re-learn these)

1. **Camera head AdaLN amplification** — CameraCausalHead's 4-iteration AdaLN
   refinement amplifies any F16 exposure on the camera-token path ~500x into
   the pose output (the q8 force-F32 precedent). Symptom: depth fine, pose
   diverges from the second streaming frame. The trunk therefore keeps an F32
   cache in every mode, and the PyTorch mirror's `CausalAttention` has no cache
   rounding at all. The amplification acts on whatever the trunk delivers:
   the q8mix experiment (f16 camera_head, q8 elsewhere) reproduced the full-q8
   pose cost because the trunk's rounding perturbs the camera tokens before
   the head ever runs — fixing the head's weights alone does not help.
2. **Post-softmax noise compounds; pre-softmax dissipates** — F16 K/V and
   Q→F16 rounding are benign end to end (proven by mirror A/Bs); P rounding,
   PV-accumulator precision and output staging are amplified by the cache
   feedback loop. Judge noise classes before optimizing kernels.
3. **gallocr stale memory on cross-size re-reserve** — backend-independent
   (CPU/CUDA/Vulkan), nondeterministic; blamed historically on "Vulkan
   strided-b mul_mat". Never re-reserve one gallocr across different graph
   sizes (`tools/mul_mat_strided_b_test.cpp` reproduces).
4. **ggml v0.21 env-only toggles** — `GGML_VK_DISABLE_*` are read inside ggml
   at device init; our options must `setenv` before
   `ggml_backend_load_all()`. `GGML_VK_FA_COOPMAT_ONLY` clears the GEMM-side
   coopmat flags while `coopmat1_fa_support` (set earlier) keeps FA on
   coopmat1.
5. **NV coopmat2 f32 mma is TF32-class** — 10-bit mantissa on inputs; same
   accuracy class as F16 for the softmax weights. Do not use it where
   post-softmax precision matters.
6. **glslang traps** — `clamp` is ambiguous when the accumulator and bound
   types differ (wrap bounds in `O_TYPE(...)`); any shader constant typed
   `FLOAT_TYPE`/`FLOAT_TYPEV4` must switch to `O_TYPE`/`O_TYPEV4` together
   with the O_TYPE change or FA_F32ACC variants fail to compile.
7. **Vulkan single-buffer limit** — S_k=286000-sized *input* tensors exceed it
   (device abort); size harness graphs so the large tensors are *outputs*.
8. **Pose convention** — decoded pose is camera-from-world; c2w sidecars must
   emit the inverse (R^T, -R^T t). The scalar-last quaternion order is
   (x, y, z, w). Both were silent bug sources.
9. **f32 patch-replay discipline** — never edit the submodule without
   regenerating + round-tripping the patch (see Patch Management); clean
   clones otherwise silently lose the change.
10. **Environment bridge must be idempotent** — ggml reads its `GGML_VK_*`
   toggles once at device init; leaving any toggle one-sided (`setenv` only)
   makes the target state depend on the call sequence, silently disabling
   coopmat flash after an in-process strict init. Assign every toggle on
   every init (set *or* unset). `setenv`/`unsetenv` are POSIX: the supported
   build target is Linux — there is no Windows build configuration in this
   repository, and adding one would also need a `_putenv`-based bridge
   (note `_putenv_s(k, "")` does *not* delete a variable; `_putenv("K=")`
   does).
11. **Keyframe skips freeze the whole cache state, not just the append** —
   a non-keyframe must not pre-evict, not append, not emit specials, and the
   next keyframe must reuse the temporal slot it would have had (official
   `total_frames_processed` does not advance; special-token tables are a
   2-row first/stream table and RoPE is purely spatial, so nothing else may
   advance either). Pre-evicting on a skip frame silently shifts the window
   and diverges from `inference_streaming`.
12. **Absolute depth gates are depth-scale-weighted** — the same engine delta
   reads as RMSE ~1e-04 but max_abs up to ~2e-02 wherever the F16-cache
   scatter tail lands (measured on 100-frame courthouse: interval=1 max_abs
   1.63e-02, interval=2 2.72e-02, same class); always pair RMSE/REL with the
   absolute view when gating or attributing (compare_parity's depth max_abs
   gate trips by design on such scenes; use its RMSE columns and the
   reconstruction RMSE gates).
13. **Parity references must match the cache contract being gated** — a flash
   row measured against an F32-cache mirror mixes the F16 K/V contract cost
   into the engine number (pure-PyTorch-measurable: pose 2.38e-03 over the
   1050-frame drive stream, first50 2.33e-05 -> lastQ 3.72e-03, §12.1 of
   validation_report.md). Gate flash rows against the
   `LINGBOT_KV_CACHE_F16=flash` mirror (strict rows: `=1`), or state the
   F32-cache reference as a known upper bound. The attribution method that
   decided §12: hold the weights constant, vary only the precision knob
   under test, then compare engine-vs-checkpoint totals against
   mirror-vs-checkpoint totals.
14. **Per-tensor sensitivity does not predict end-to-end quantization cost** —
   the q8mix experiment (q8 trunk+depth_head, f16 camera_head) landed in the
   full-q8 pose class (3.0e-02) although the top-ranked tensor
   (camera_head.pose_branch.fc2, 0.84% rel err) was kept in f16; the trunk's
   collective q8 rounding dominates (§12.2). Mixed-precision q8 is closed as
   a negative result; reducing the q8 pose cost requires an f16/f32 trunk
   (= the f16 GGUF).

## Documentation Map

- `cpp_ggml/benchmarks/validation_report.md` — final measured matrix,
  evidence charts, capacity/latency tables (this repo's source of truth for
  numbers).
- `cpp_ggml/models/MODEL_CARD.md` — GGUF formats, accuracy contracts,
  memory limits, skyseg companion models.
- `cpp_ggml/benchmarks/skyseg_ggml.md` — native skyseg runtime details.
- `cpp_ggml/README.md` — runtime internals, scripts, parity methodology.
- Root `README.md` — product overview, quick start, GUI.
- Git history — all intermediate experiment logs (attribution tables, coopmat
  probes, per-frame decompositions) removed from the docs on 2026-09-14.
