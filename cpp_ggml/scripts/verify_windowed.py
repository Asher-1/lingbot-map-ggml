#!/usr/bin/env python3
"""Windowed-mode verification: GGML windowed orchestration vs the official
PyTorch ``inference_windowed`` on the same frames.

The windowed mode of the GGML chain (ggml_demo.py --mode windowed) reuses the
validated streaming primitive per window and adds only the orchestration:
window splitting, pairwise similarity alignment on the overlap, and the
de-duplicating stitch. This script isolates exactly those pieces:

1. per-window RAW outputs (pose_enc / depth) — engine parity per window
   against the official ``lingbot-map.pt`` checkpoint (carries the documented
   f16 weight-format noise, ~1.7e-04 pose over the 286-frame stream);
2. the merged (aligned + stitched) reconstruction — orchestration parity.
   Two levels: a cross-warp check (the torch-side alignment transforms
   applied to the ggml raw windows — with identical alignment inputs this
   isolates the warp + stitch implementation; must hold raw-parity class)
   and the self-aligned merge (each side estimates its own (s, R, t) from
   its own overlap depths; the independent depth-ratio medians differ by
   the method's own ~1e-2 noise floor, so this level is a loose sanity
   gate only).

Run with a torch-capable interpreter (it also drives the native CLI):

    ~/anaconda3/envs/python3.12/bin/python cpp_ggml/scripts/verify_windowed.py \
        --frames 24 --window-size 10 --overlap-size 4

Gates: raw pose/depth 1e-3/3e-3, cross-warp 3e-3 (all the run_reconstruction
tolerance class), self-merged 5e-2 sanity.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_ggml" / "scripts"))

import ggml_demo  # noqa: E402  (numpy/PIL only — no torch import at module level)


def load_frames(scene: Path, frames: int, width: int):
    files = sorted(scene.glob("*.png"))[:frames]
    if not files:
        raise SystemExit(f"no PNG frames in {scene}")
    src_w, src_h = Image.open(files[0]).size
    height = round(src_h * (width / src_w) / 14) * 14
    images = np.stack([
        np.asarray(Image.open(p).convert("RGB").resize((width, height), Image.Resampling.BICUBIC),
                   dtype=np.float32).transpose(2, 0, 1) / 255.0
        for p in files])
    return images, height, width


def rmse(a, b):
    a = np.squeeze(np.asarray(a, np.float64))  # torch depth carries a trailing (..., 1)
    b = np.squeeze(np.asarray(b, np.float64))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=ROOT / "example/courthouse")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--overlap-size", type=int, default=4)
    ap.add_argument("--num-scale-frames", type=int, default=8)
    ap.add_argument("--kv-cache-scale", type=int, default=8)
    ap.add_argument("--kv-cache-window", type=int, default=64)
    ap.add_argument("--gguf", type=Path, default=ROOT / "cpp_ggml/models/gguf/lingbot-map-f16.gguf")
    ap.add_argument("--build", type=Path, default=ROOT / "cpp_ggml/build-cuda")
    ap.add_argument("--backend", default="CUDA0")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "cpp_ggml/models/pytorch/lingbot-map.pt")
    ap.add_argument("--mirror-gguf", type=Path, default=None,
                    help="decode this GGUF into the torch windowed reference instead of "
                         "--ckpt: with the same GGUF on both sides the raw/cross-warp "
                         "levels become pure engine parity (weight loss cancels); the "
                         "--ckpt comparison additionally carries the weight-format noise")
    ap.add_argument("--width", type=int, default=518)
    args = ap.parse_args()

    images, height, width = load_frames(args.scene, args.frames, args.width)
    S = images.shape[0]
    ws = max(1, min(args.num_scale_frames, S))
    eff_overlap = min(args.overlap_size, S - 1) if S > 1 else 0
    eff_window = min(ws + max(args.window_size - ws, 0), S)
    windows = ggml_demo._split_windows(S, eff_window, eff_overlap)
    print(f"frames={S} grid={width}x{height} windows={windows} "
          f"(window={args.window_size}, overlap={eff_overlap}, scale={ws})")

    # ------------------------------------------------------------------
    # GGML side: per-window raw runs (fresh CLI process == fresh KV cache),
    # then the same alignment + stitch helpers run_ggml_windowed uses.
    # ------------------------------------------------------------------
    ggml_args = SimpleNamespace(
        model=str(args.gguf), build=args.build, backend=args.backend,
        kv_cache_scale=args.kv_cache_scale, kv_cache_window=args.kv_cache_window,
        num_scale_frames=args.num_scale_frames, mask_sky=False,
        skyseg="", sky_mask_dir=None, sky_mask_visualization_dir=None)
    ggml_raw, ggml_warped = [], []
    for wi, (start, end) in enumerate(windows):
        pose_enc, depth, c2w, intr, conf = ggml_demo.run_ggml_inference(
            ggml_args, images[start:end], height, width)
        win = {"pose_enc": pose_enc, "depth": depth, "depth_conf": conf,
               "c2w": c2w, "intrinsics": intr}
        ggml_raw.append(win)
        if wi > 0:
            s_ab, R_ab, t_ab = ggml_demo._pairwise_alignment(ggml_warped[-1], win, eff_overlap)
            print(f"[ggml window {wi + 1}] scale={s_ab:.6f} t=[{t_ab[0]:.4f} {t_ab[1]:.4f} {t_ab[2]:.4f}]")
            win = ggml_demo._warp_window(win, s_ab, R_ab, t_ab)
        ggml_warped.append(win)
    ggml_merged = ggml_demo._stitch_windows(ggml_warped, eff_overlap)

    # ------------------------------------------------------------------
    # PyTorch side: replicate inference_windowed's fixed-interval loop for
    # the raw per-window predictions, then call the model's own aligner.
    # ------------------------------------------------------------------
    from lingbot_map.models.gct_stream_window import GCTStream
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mirror-contract numerics (run_pytorch_reference.py): FP32 weights, no
    # autocast, TF32/cuDNN off — bf16 self-noise would bury the engine parity
    # this gate isolates (the official demo's bf16 deployment noise is a
    # separate, documented factor).
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.enabled = False
    # enable_3d_rope=False matches the mirror contract (run_pytorch_reference.py):
    # the GGML graph implements the 2D spatial RoPE path, so the reference must
    # not enable the deployment-only 3D temporal RoPE.
    model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=False, max_frame_num=1024,
                      kv_cache_sliding_window=args.kv_cache_window,
                      kv_cache_scale_frames=args.num_scale_frames,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                      use_sdpa=True, camera_num_iterations=4)
    if args.mirror_gguf:
        from run_pytorch_reference import load_gguf_state
        state = load_gguf_state(args.mirror_gguf)
        reference = f"mirror {args.mirror_gguf.name}"
    else:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        reference = f"checkpoint {args.ckpt.name}"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"reference mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    print(f"torch reference: {reference}")

    def collect(out, lists):
        lists["pose_enc"].append(out["pose_enc"].detach())
        lists["depth"].append(out["depth"].detach())
        lists["depth_conf"].append(out["depth_conf"].detach())

    torch_raw, torch_warped, torch_transforms = [], [], []
    t_images = torch.from_numpy(images[None]).to(device)
    with torch.no_grad():
        for wi, (start, end) in enumerate(windows):
            model.clean_kv_cache()
            win_len = end - start
            window_scale = min(ws, win_len)
            scale_out = model.forward(t_images[:, start:start + window_scale],
                                      num_frame_for_scale=window_scale,
                                      num_frame_per_block=window_scale,
                                      causal_inference=True)
            lists = {"pose_enc": [], "depth": [], "depth_conf": []}
            collect(scale_out, lists)
            del scale_out
            for i in range(window_scale, win_len):
                frame_out = model.forward(t_images[:, start + i:start + i + 1],
                                          num_frame_for_scale=window_scale,
                                          num_frame_per_block=1, causal_inference=True)
                collect(frame_out, lists)
                del frame_out
            win = {"pose_enc": torch.cat(lists["pose_enc"], dim=1),
                   "depth": torch.cat(lists["depth"], dim=1),
                   "depth_conf": torch.cat(lists["depth_conf"], dim=1)}
            raw = {k: v[0].float().cpu().numpy() for k, v in win.items()}
            torch_raw.append(raw)
            if wi > 0:
                s_rel, R_rel, t_rel = model._pairwise_alignment(
                    torch_warped[-1], win, eff_overlap, 1, device, torch.float32)
                torch_transforms.append((float(s_rel[0]), R_rel[0].float().cpu().numpy(),
                                         t_rel[0].float().cpu().numpy()))
                print(f"[torch window {wi + 1}] scale={float(s_rel[0]):.6f} "
                      f"t=[{float(t_rel[0, 0]):.4f} {float(t_rel[0, 1]):.4f} {float(t_rel[0, 2]):.4f}]")
                warped = model._warp_predictions(win, R_rel, t_rel, s_rel, 1)
            else:
                warped = win
            torch_warped.append(warped)
        model._last_window_size = eff_window
        model._last_overlap_size = eff_overlap
        merged = model._align_and_stitch_windows(torch_warped, scale_mode="median")
    torch_merged = {k: merged[k][0].float().cpu().numpy() for k in ("pose_enc", "depth", "depth_conf")}

    def decode_c2w(pose_enc):
        pe = torch.from_numpy(pose_enc)[None].to(device)
        with torch.no_grad():
            extrinsic, _ = pose_encoding_to_extri_intri(pe, (height, width))
            e4 = torch.zeros(1, extrinsic.shape[1], 4, 4, device=extrinsic.device, dtype=extrinsic.dtype)
            e4[..., :3, :4] = extrinsic
            e4[..., 3, 3] = 1.0
            e4 = closed_form_inverse_se3_general(e4)
        return e4[0, 0].float().cpu().numpy()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\n=== per-window RAW parity (engine [+ weight noise]) ===")
    raw_pose = raw_depth = 0.0
    for wi, (start, end) in enumerate(windows):
        rp = rmse(ggml_raw[wi]["pose_enc"], torch_raw[wi]["pose_enc"])
        rd = rmse(ggml_raw[wi]["depth"], torch_raw[wi]["depth"])
        raw_pose, raw_depth = max(raw_pose, rp), max(raw_depth, rd)
        print(f"window {wi + 1} [{start},{end}): pose_enc rmse={rp:.3e} depth rmse={rd:.3e}")
    print(f"raw worst: pose={raw_pose:.3e} depth={raw_depth:.3e}")

    # Cross-warp stitch: apply the TORCH-side alignment transforms to the
    # GGML raw windows, then stitch with the ggml_demo helpers. With the
    # alignment inputs held identical this isolates the warp + stitch
    # implementation from the method's own scale-freedom noise.
    print("\n=== cross-warp stitch (torch alignment applied to ggml raw) ===")
    ggml_cross = [ggml_raw[0]]
    for wi in range(1, len(windows)):
        s_ab, R_ab, t_ab = torch_transforms[wi - 1]
        ggml_cross.append(ggml_demo._warp_window(ggml_raw[wi], s_ab, R_ab, t_ab))
    ggml_cross_merged = ggml_demo._stitch_windows(ggml_cross, eff_overlap)
    x_pose = rmse(ggml_cross_merged["pose_enc"], torch_merged["pose_enc"])
    x_depth = rmse(ggml_cross_merged["depth"], torch_merged["depth"])
    x_c2w = np.stack([decode_c2w(pe) for pe in torch_merged["pose_enc"]])
    x_center = rmse(ggml_cross_merged["c2w"][:, :3, 3], x_c2w[:, :3, 3])
    x_rot = rmse(ggml_cross_merged["c2w"][:, :3, :3], x_c2w[:, :3, :3])
    print(f"cross-warp: pose_enc rmse={x_pose:.3e} depth rmse={x_depth:.3e} "
          f"c2w center rmse={x_center:.3e} c2w rotation rmse={x_rot:.3e}")

    print("\n=== self-aligned merged (each side's own alignment: method noise) ===")
    m_pose = rmse(ggml_merged["pose_enc"], torch_merged["pose_enc"])
    m_depth = rmse(ggml_merged["depth"], torch_merged["depth"])
    # The independent depth-ratio scale estimates differ by the method's own
    # ~1-2% (the overlap frames are predicted under different attention
    # contexts: prev-window streaming vs curr-window scale pass), so the
    # merged depth error is a scene-relative sanity signal, not an
    # orchestration check — the cross-warp level above is the decisive one.
    m_depth_rel = m_depth / float(np.mean(np.abs(torch_merged["depth"])))
    t_c2w = np.stack([decode_c2w(pe) for pe in torch_merged["pose_enc"]])
    g_c2w = ggml_merged["c2w"]
    m_center = rmse(g_c2w[:, :3, 3], t_c2w[:, :3, 3])
    m_rot = rmse(g_c2w[:, :3, :3], t_c2w[:, :3, :3])
    print(f"merged frames: ggml={ggml_merged['pose_enc'].shape[0]} torch={torch_merged['pose_enc'].shape[0]}")
    print(f"merged: pose_enc rmse={m_pose:.3e} depth rmse={m_depth:.3e} "
          f"(relative {m_depth_rel:.3e}) c2w center rmse={m_center:.3e} "
          f"c2w rotation rmse={m_rot:.3e}")

    ok = (raw_pose <= 1e-3 and raw_depth <= 3e-3
          and x_pose <= 3e-3 and x_depth <= 3e-3
          and x_center <= 3e-3 and x_rot <= 3e-3
          and m_pose <= 5e-2 and m_depth_rel <= 5e-2 and m_center <= 5e-2)
    print(f"\nWINDOWED {'PASS' if ok else 'FAIL'} "
          f"(gates: raw 1e-3/3e-3, cross-warp 3e-3, self-merged sanity "
          f"pose/center 5e-2 + depth relative 5e-2)")
    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
