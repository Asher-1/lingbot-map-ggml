#!/usr/bin/env python3
"""Summarize a full-stream LBO against a reference NPZ into the standard CSV."""
import argparse, csv, struct
import numpy as np

def read_lbo(path):
    raw = open(path, 'rb').read()
    magic, n_pose, n_depth = struct.unpack('<III', raw[:12])
    if magic != 0x4c424f31: raise RuntimeError('invalid LBO1 output')
    v = np.frombuffer(raw, dtype=np.float32, offset=12)
    return v[:n_pose], v[n_pose:n_pose+n_depth]

ap = argparse.ArgumentParser()
ap.add_argument('lbo')
ap.add_argument('reference')
ap.add_argument('out')
ap.add_argument('--engine', default='ggml-q8-f16cache-cuda0')
ap.add_argument('--scene', default='courthouse')
args = ap.parse_args()

pose, depth = read_lbo(args.lbo)
ref = np.load(args.reference)
rpose = ref['pose_enc'].reshape(-1, 9).astype(np.float64)
rdepth = ref['depth'].reshape(-1).astype(np.float64)
if pose.size != rpose.size or depth.size != rdepth.size:
    raise SystemExit(f'shape mismatch: pose {pose.size} vs {rpose.size}, depth {depth.size} vs {rdepth.size}')
pose_err = pose.astype(np.float64).reshape(-1, 9) - rpose
depth_err = depth.astype(np.float64) - rdepth
ate = float(np.sqrt(np.mean(pose_err * pose_err)))
drmse = float(np.sqrt(np.mean(depth_err * depth_err)))
pose_max = float(np.abs(pose_err).max())
depth_max = float(np.abs(depth_err).max())
viol_p = int((np.abs(pose_err) > 1e-3 + 1e-3*np.abs(rpose)).sum())
viol_d = int((np.abs(depth_err) > 1e-3 + 1e-3*np.abs(rdepth)).sum())
with open(args.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['engine','scene','frames','ate_rmse','depth_rmse',
                                      'pose_max_abs','depth_max_abs','allclose_pose_viol','allclose_depth_viol'],
                       lineterminator='\n')
    w.writeheader()
    w.writerow({'engine': args.engine, 'scene': args.scene,
                'frames': pose.size // 9, 'ate_rmse': ate, 'depth_rmse': drmse,
                'pose_max_abs': pose_max, 'depth_max_abs': depth_max,
                'allclose_pose_viol': viol_p, 'allclose_depth_viol': viol_d})
    w.writerow({'engine': 'pytorch-reference', 'scene': args.scene, 'frames': pose.size // 9,
                'ate_rmse': 0.0, 'depth_rmse': 0.0, 'pose_max_abs': 0.0, 'depth_max_abs': 0.0,
                'allclose_pose_viol': 0, 'allclose_depth_viol': 0})
print(f'{args.engine}: pose_rmse={ate:.6g} depth_rmse={drmse:.6g} pose_max={pose_max:.3g} '
      f'depth_max={depth_max:.3g} viol_p={viol_p} viol_d={viol_d}')
