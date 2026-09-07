"""End-to-end verification of the run_gui.sh GGML chain vs the official pipeline.

Reuses ggml_demo's exact code path (load_official_frames -> run_ggml_inference
-> build_pred_dict — the same functions the GUI runs) and compares against the
official PyTorch streaming reference (fp32 checkpoint) and the
cache-precision-aware mirror:
  localization   : pose_enc RMSE
  trajectory     : camera-center deviation (m) via the official pose decoder
  mapping        : depth RMSE
  reconstruction : official unproject -> per-point distance (conf>1.5 viewer rule)
  intrinsics     : fx/fy/cx/cy RMSE

Usage:
  python3 cpp_ggml/scripts/verify_gui_chain.py [--frames N]
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_ggml" / "scripts"))
import ggml_demo  # noqa: E402  (the run_gui.sh engine module)


def decode_c2w(pose_enc, hw):
    """Official decode -> postprocess convention (camera-from-world -> c2w)."""
    import torch
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    ext, intr = pose_encoding_to_extri_intri(
        torch.from_numpy(pose_enc.astype(np.float32))[None], image_size_hw=hw)
    ext = ext[0].numpy().astype(np.float64)   # (S,4,4) camera-from-world
    c2w = ext.copy()
    R, t = ext[:, :3, :3], ext[:, :3, 3:]
    c2w[:, :3, :3] = np.transpose(R, (0, 2, 1))
    c2w[:, :3, 3:4] = -np.transpose(R, (0, 2, 1)) @ t
    return c2w, intr[0].numpy().astype(np.float64)


def rmse(a, b):
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--reference", default=str(ROOT / "cpp_ggml/benchmarks/official_fp32_286.npz"))
    ap.add_argument("--mirror", default=str(ROOT / "cpp_ggml/benchmarks/mirror_qf16_286.npz"))
    args_cli = ap.parse_args()
    N = args_cli.frames

    import torch
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

    ref = np.load(args_cli.reference)
    mir = np.load(args_cli.mirror)
    hw = (ref["depth"].shape[-2], ref["depth"].shape[-1])
    ref_pose = ref["pose_enc"].reshape(-1, 9)[:N]
    ref_depth = ref["depth"].reshape(-1, *hw)[:N]
    ref_conf = ref["depth_conf"].reshape(-1, *hw)[:N]
    m_pose = mir["pose_enc"].reshape(-1, 9)[:N]
    m_depth = mir["depth"].reshape(-1, *hw)[:N]

    print(f"reference: {args_cli.reference} (official fp32 .pt, {ref['depth'].shape[1]} frames), first {N}")
    for backend, build, tag in [("CUDA0", ROOT / "cpp_ggml/build-cuda", "CUDA"),
                                ("Vulkan0", ROOT / "cpp_ggml/build-vulkan", "Vulkan")]:
        for mode, env_flag in [("strict", "1"), ("flash", "flash")]:
            os.environ["LINGBOT_KV_CACHE_F16"] = env_flag
            args = SimpleNamespace(
                build=build, model=f"{ROOT}/cpp_ggml/models/gguf/lingbot-map-f16.gguf",
                backend=backend, mask_sky=False,
                skyseg=f"{ROOT}/cpp_ggml/models/gguf/lingbot-map-skyseg-f16.gguf",
                sky_mask_dir=None, sky_mask_visualization_dir=None,
                kv_cache_scale=8, kv_cache_window=64, num_scale_frames=8)
            files = sorted(glob.glob(f"{ROOT}/example/courthouse/*.png"))[:N]
            assert len(files) == N
            images = ggml_demo.load_official_frames(files, 518, 14)
            height, width = int(images.shape[2]), int(images.shape[3])
            pose_enc, depth, c2w, intr, conf = ggml_demo.run_ggml_inference(
                args, images, height, width, on_frame=None)
            S = pose_enc.shape[0]
            c2w_ref, intr_ref = decode_c2w(ref_pose[:S], hw)   # intr_ref: (S,3,3)
            intr_ref4 = np.stack([intr_ref[:, 0, 0], intr_ref[:, 1, 1],
                                  intr_ref[:, 0, 2], intr_ref[:, 1, 2]], axis=1)

            pose_r = rmse(pose_enc, ref_pose[:S])
            cc_g = c2w[:S, :3, 3].astype(np.float64)
            cc_r = c2w_ref[:S, :3, 3]
            cc = np.linalg.norm(cc_g - cc_r, axis=1)
            depth_r = rmse(depth.reshape(S, -1), ref_depth.reshape(S, -1))
            intr_r = rmse(intr, intr_ref4)

            intr_g3 = np.zeros((S, 3, 3)); intr_g3[:, 0, 0] = intr[:, 0]
            intr_g3[:, 1, 1] = intr[:, 1]; intr_g3[:, 0, 2] = intr[:, 2]
            intr_g3[:, 1, 2] = intr[:, 3]; intr_g3[:, 2, 2] = 1.0
            ir = intr_ref4
            intr_r3 = np.zeros((S, 3, 3)); intr_r3[:, 0, 0] = ir[:, 0]
            intr_r3[:, 1, 1] = ir[:, 1]; intr_r3[:, 0, 2] = ir[:, 2]
            intr_r3[:, 1, 2] = ir[:, 3]; intr_r3[:, 2, 2] = 1.0
            pts_g = unproject_depth_map_to_point_map(
                depth.reshape(S, *hw, 1), c2w[:S, :3, :4].astype(np.float32),
                intr_g3.astype(np.float32))
            pts_r = unproject_depth_map_to_point_map(
                ref_depth.reshape(S, *hw, 1), c2w_ref[:S, :3, :4].astype(np.float32),
                intr_r3.astype(np.float32))
            mask = (conf.reshape(S, *hw) > 1.5) & (ref_conf > 1.5)
            d = np.linalg.norm(pts_g[mask] - pts_r[mask], axis=1).astype(np.float64)

            print(f"-- {tag} {mode} ({N} frames, run_gui.sh chain, f16 GGUF)")
            print(f"   localization  pose_enc RMSE        : {pose_r:.3e}")
            print(f"   trajectory    center dev med/max  : {np.median(cc):.3e} / {cc.max():.3e} m  (track extent ~4 m)")
            print(f"   mapping       depth RMSE          : {depth_r:.3e}")
            print(f"   reconstruction point dev med/mean: {np.median(d):.3e} / {np.mean(d):.3e} m  (conf>1.5, {mask.sum()} pts)")
            print(f"   intrinsics    RMSE                : {intr_r:.3e}")
            print(f"   vs cache-aware mirror: pose {rmse(pose_enc, m_pose[:S]):.3e}"
                  f"  depth {rmse(depth.reshape(S, -1), m_depth[:S].reshape(S, -1)):.3e}")


if __name__ == "__main__":
    main()
