#!/usr/bin/env python3
"""Run a real scene slice through GGML and PyTorch and compare reconstruction outputs."""
import argparse, csv, struct, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

# Child scripts need torch and the repo on sys.path, so reuse the interpreter
# this script was launched with rather than whatever "python3" resolves to.
PYTHON = sys.executable

def read_lbo(path):
    raw = Path(path).read_bytes(); magic, np_, nd = struct.unpack('<III', raw[:12])
    if magic != 0x4c424f31: raise RuntimeError('invalid LBO1 output')
    vals = np.frombuffer(raw, dtype=np.float32, offset=12)
    return vals[:np_], vals[np_:np_+nd]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=Path('cpp_ggml/models/gguf/lingbot-map-q8.gguf'))
    ap.add_argument('--build', type=Path, default=Path('/tmp/lingbot-cuda-build'))
    ap.add_argument('--backend', default='CUDA0')
    ap.add_argument('--scene', type=Path, default=Path('example/courthouse'))
    ap.add_argument('--frames', type=int, default=2)
    ap.add_argument('--kv-cache-scale', type=int, default=1)
    ap.add_argument('--kv-cache-window', type=int, default=4)
    ap.add_argument('--out', type=Path, default=Path('cpp_ggml/benchmarks/reconstruction_compare.csv'))
    ap.add_argument('--effect-out', type=Path, default=Path('cpp_ggml/benchmarks/reconstruction_effect.png'))
    ap.add_argument('--full-export-prefix', type=Path,
                    help='write all-frame PyTorch/GGML colored PLYs and a point-cloud comparison')
    ap.add_argument('--reference-gguf', type=Path,
                    help='decode this GGUF into the independent PyTorch reference; isolates GGML graph parity')
    ap.add_argument('--pose-tol', type=float, default=1e-3)
    ap.add_argument('--depth-tol', type=float, default=3e-3)
    ap.add_argument('--height', type=int,
                    help='input height; defaults to the official crop rule '
                         '(round(h * width / w / 14) * 14), which preserves the source '
                         'aspect ratio. Force a value only to exercise DINO positional '
                         'interpolation at a deliberately distorted aspect')
    ap.add_argument('--width', type=int, default=518)
    ap.add_argument('--patch-size', type=int, default=14)
    args = ap.parse_args()
    model_label = next((label for label in ('f32', 'f16', 'q8', 'q4')
                        if label in args.model.stem.lower()), 'ggml')
    files = sorted(args.scene.glob('*.png'))[:args.frames]
    if not files: raise SystemExit(f'no PNG frames in {args.scene}')
    w = args.width
    if args.height:
        h = args.height
    else:
        # lingbot_map/utils/load_fn.py load_and_preprocess_images(mode="crop"):
        # resize to the target width, snap the height to the patch grid. Stretching
        # to a square instead costs real accuracy (the model's own depth confidence
        # drops), so aspect-preserving is the default.
        src_w, src_h = Image.open(files[0]).size
        h = round(src_h * (w / src_w) / args.patch_size) * args.patch_size
    frames = np.stack([np.asarray(Image.open(p).convert('RGB').resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32).transpose(2, 0, 1) / 255.0 for p in files])
    np.save('/tmp/lingbot_scene.npy', frames[None])
    frames.tofile('/tmp/lingbot_scene.bin')
    lbo = Path('/tmp/lingbot_scene.lbo')
    cli = args.build / 'lingbot-map-cli'
    env = dict(__import__('os').environ,
               LINGBOT_KV_CACHE_SCALE=str(args.kv_cache_scale),
               LINGBOT_KV_CACHE_WINDOW=str(args.kv_cache_window),
               LINGBOT_NUM_SCALE_FRAMES=str(args.kv_cache_scale))
    subprocess.run([str(cli), str(args.model), '/tmp/lingbot_scene.bin', args.backend, str(h), str(w), str(lbo), str(len(files))], check=True, env=env)
    cpp_pose, cpp_depth = read_lbo(lbo)
    ref = Path('/tmp/lingbot_scene_ref.npz')
    reference_cmd = [PYTHON, str(Path(__file__).with_name('run_pytorch_reference.py')), 'cpp_ggml/models/pytorch/lingbot-map.pt', '/tmp/lingbot_scene.npy', str(ref), '--device', 'cuda', '--streaming', '--scale-frames', str(args.kv_cache_scale), '--kv-cache-scale', str(args.kv_cache_scale), '--kv-cache-window', str(args.kv_cache_window)]
    if args.reference_gguf:
        reference_cmd.extend(['--gguf', str(args.reference_gguf)])
    subprocess.run(reference_cmd, check=True)
    torch_ref = np.load(ref)
    pose = torch_ref['pose_enc'].reshape(-1)
    depth = torch_ref['depth'].reshape(-1)
    if cpp_pose.shape != pose.shape or cpp_depth.shape != depth.shape: raise SystemExit('shape mismatch')
    pose_err = cpp_pose.reshape(-1, 9) - pose.reshape(-1, 9)
    depth_err = cpp_depth - depth
    ate = float(np.sqrt(np.mean(pose_err * pose_err)))
    drmse = float(np.sqrt(np.mean(depth_err * depth_err)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=['engine','scene','ate_rmse','depth_rmse'], lineterminator='\n'); wri.writeheader()
        wri.writerow({'engine':f'ggml-{model_label}-{args.backend.lower()}', 'scene':args.scene.name, 'ate_rmse':ate, 'depth_rmse':drmse})
        wri.writerow({'engine':'pytorch-reference', 'scene':args.scene.name, 'ate_rmse':0.0, 'depth_rmse':0.0})
    subprocess.run([PYTHON, str(Path(__file__).with_name('plot_reconstruction.py')), str(args.out)], check=True)
    subprocess.run([
        PYTHON, str(Path(__file__).with_name('plot_reconstruction_effect.py')),
        '--scene', str(args.scene), '--pytorch', str(ref), '--ggml-lbo', str(lbo),
        '--ggml-label', model_label, '--frames', str(len(files)), '--output', str(args.effect_out),
        '--height', str(h), '--width', str(w),
    ], check=True)
    if args.full_export_prefix:
        subprocess.run([
            PYTHON, str(Path(__file__).with_name('export_full_reconstruction.py')),
            '--scene', str(args.scene), '--pytorch', str(ref), '--ggml-lbo', str(lbo),
            '--frames', str(len(files)), '--out-prefix', str(args.full_export_prefix),
            '--ggml-label', model_label, '--height', str(h), '--width', str(w),
        ], check=True)
    ok = ate <= args.pose_tol and drmse <= args.depth_tol
    # The tolerances are calibrated for graph parity, i.e. against a reference
    # that decodes the same GGUF. Without --reference-gguf the comparison also
    # carries the weight format's own loss (~20-30x larger for q8), so name the
    # reference in the verdict instead of reporting a bare FAIL.
    reference = 'gguf-decoded' if args.reference_gguf else 'fp32-checkpoint (includes weight-format loss)'
    print(f'RECONSTRUCTION {"PASS" if ok else "FAIL"} scene={args.scene.name} frames={len(files)} grid={w}x{h} reference={reference} pose_rmse={ate:.6g} depth_rmse={drmse:.6g}')
    if not ok: raise SystemExit(3)

if __name__ == '__main__': main()
