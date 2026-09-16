#!/usr/bin/env python3
"""One-command GGML reconstruction GUI, matching the official demo.py viewer.

Runs the native C++ GGML streaming runtime on an image folder while a live
viser scene grows frame by frame (point clouds + camera frustums + progress),
then hands the finished reconstruction to the exact same PointCloudViewer the
official Python demo uses, so the final 3D visualization (point clouds,
camera frustums, trajectory, playback) is identical — only the inference
backend differs.

Preprocessing and the streaming profile mirror `demo.py` as well: frames go
through the official crop rule (width -> --image_size, height snapped to the
patch grid, center-cropped if taller than the target) and the upstream
`scale=8/window=64` cache profile with the persistent F16 KV cache. That is
the exact configuration the 286-frame parity/reconstruction evidence in
`cpp_ggml/benchmarks/` was measured at.

Inference modes mirror `demo.py --mode` too:
  - streaming (default): frame-by-frame with the persistent KV cache.
  - windowed: overlapping windows for long sequences — each window runs the
    streaming primitive above with a fresh KV cache (a fresh CLI process),
    then consecutive windows are similarity-aligned on the overlap and
    stitched with the de-duplicating slice table, matching
    `GCTStream.inference_windowed` (keyframe_interval=1, the demo.py
    windowed default; the native engine streams every frame).

Example:
    python3 ggml_demo.py --image_folder example/courthouse --frames 40
    python3 ggml_demo.py --image_folder example/courthouse --mode windowed \
        --window_size 64 --overlap_size 16
"""
import argparse
import math
import os
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_LBO1 = 0x4C424F31
_LBP2 = 0x4C425032
_LBF3 = 0x4C424633  # per-frame streaming record emitted by lingbot-map-cli


def read_lbo(path: Path, frames: int, height: int, width: int):
    raw = path.read_bytes()
    magic, n_pose, n_depth = struct.unpack("<III", raw[:12])
    if magic != _LBO1 or n_pose != frames * 9 or n_depth != frames * height * width:
        raise SystemExit(f"invalid GGML LBO1 output: {path}")
    vals = np.frombuffer(raw, dtype=np.float32, offset=12)
    return vals[:n_pose].reshape(frames, 9), vals[n_pose:].reshape(frames, height, width)


def read_postprocess(path: Path, frames: int):
    """LBP2: c2w matrices, intrinsics [fx, fy, cx, cy], depth confidence."""
    raw = path.read_bytes()
    magic, n_c2w, n_k = struct.unpack("<III", raw[:12])
    if magic != _LBP2 or len(raw) < 16:
        raise SystemExit(f"invalid GGML LBP2 postprocess output: {path}")
    n_conf, = struct.unpack("<I", raw[12:16])
    vals = np.frombuffer(raw, dtype=np.float32, offset=16)
    if n_c2w != frames * 16 or n_k != frames * 4 or vals.size != n_c2w + n_k + n_conf:
        raise SystemExit(f"truncated GGML LBP2 postprocess output: {path}")
    c2w = vals[:n_c2w].reshape(frames, 4, 4).copy()
    k = vals[n_c2w:n_c2w + n_k].reshape(frames, 4).copy()
    conf = vals[n_c2w + n_k:].copy()
    return c2w, k, conf


def read_stream_frame(path: Path):
    """LBF3 per-frame record from lingbot-map-cli's stream-out directory."""
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, OSError):
        return None
    if len(raw) < 16:
        return None
    magic, idx, fh, fw = struct.unpack("<IIII", raw[:16])
    if magic != _LBF3:
        return None
    vals = np.frombuffer(raw, dtype=np.float32, offset=16)
    plane = fh * fw
    if vals.size < 9 + 2 * plane + 16 + 4:
        return None  # writer still mid-flush; skip until the FRAME line re-checks
    return {
        "index": idx,
        "height": fh,
        "width": fw,
        "pose_enc": vals[:9].copy(),
        "depth": vals[9:9 + plane].reshape(fh, fw).copy(),
        "depth_conf": vals[9 + plane:9 + 2 * plane].reshape(fh, fw).copy(),
        "c2w": vals[9 + 2 * plane:9 + 2 * plane + 16].reshape(4, 4).copy(),
        "intrinsics": vals[9 + 2 * plane + 16:9 + 2 * plane + 20].copy(),
    }


def extract_video_frames(video_path, fps=10, out_dir=None):
    """Sample a video into JPEG frames at `fps`, mirroring demo.py's loader.

    Returns (frame_paths, resolved_folder). cv2 is imported lazily so the
    image-folder path keeps working without OpenCV installed.
    """
    try:
        import cv2
    except ImportError:
        raise SystemExit("--video_path requires OpenCV: pip install opencv-python-headless")
    import tempfile
    out_dir = out_dir or tempfile.mkdtemp(prefix="lingbot_ggml_frames_")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / fps))
    saved, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
        idx += 1
    cap.release()
    print(f"Extracted {len(saved)} frames from video ({total} total, interval={interval})")
    return saved, out_dir


def load_official_frames(files, image_size, patch_size, height=None, width=None):
    """Preprocess like lingbot_map.utils.load_fn.load_and_preprocess_images(mode="crop").

    Width becomes `image_size`, the height follows the source aspect ratio snapped
    to the patch grid, and a taller-than-target height is center-cropped. Feeding
    a stretched grid instead costs real accuracy (the model's own depth confidence
    drops from 6.39 to 5.22 on courthouse), so explicit --height/--width are only
    for reproducing the deliberately distorted DINO-interpolation parity rows.

    Reimplemented on PIL rather than imported from lingbot_map so the GGML GUI
    keeps working without torch installed.
    """
    src_w, src_h = ImageOps.exif_transpose(Image.open(files[0])).size
    if width and height:
        out_w, out_h, crop = width, height, False
    else:
        out_w = width or image_size
        out_h = height or round(src_h * (out_w / src_w) / patch_size) * patch_size
        crop = height is None
    frames = []
    for path in files:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = img.resize((out_w, out_h), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if crop and out_h > image_size:
            start = (out_h - image_size) // 2
            arr = arr[:, start:start + image_size, :]
        frames.append(arr)
    return np.stack(frames)


class StreamingViewer:
    """Live viser scene that grows frame-by-frame while GGML streams.

    A lightweight preview so the reconstruction is watchable during the long
    native inference: each completed frame adds its confidence-filtered point
    cloud (depth unprojected with the frame's c2w/intrinsics, exactly the
    geometry the official viewer will show) and a camera frustum, plus
    progress / ETA text. Once inference finishes the server is stopped and the
    full official PointCloudViewer takes over on the same port.
    """

    def __init__(self, port, total_frames, image_h, image_w, conf_threshold=1.5,
                 stride=4, point_size=0.00001):
        import viser
        self._viser = viser
        self.total = total_frames
        self.conf_thr = conf_threshold
        self.stride = max(1, int(stride))
        self.point_size = point_size
        self.count = 0
        self.degenerate = 0
        self.t0 = time.time()
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        with self.server.gui.add_folder("GGML streaming reconstruction"):
            self._status = self.server.gui.add_text("Status", "warming up (scale pass)...")
            self._counter = self.server.gui.add_text("Frames", f"0 / {total_frames}")
            self._elapsed = self.server.gui.add_text("Elapsed", "0.0s")
            self._eta = self.server.gui.add_text("ETA", "--")
        self.server.gui.add_button("Fit view to reconstruction").on_click(
            lambda _: self._fit_view())
        self._last_world = None

    @staticmethod
    def _quat_wxyz(R):
        """Rotation matrix (3,3) -> unit quaternion [w, x, y, z]."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        return q / (np.linalg.norm(q) or 1.0)

    def _fit_view(self):
        if self._last_world is None or not self._last_world.size:
            return
        center = self._last_world.mean(axis=0)
        radius = float(np.linalg.norm(self._last_world - center, axis=1).max()) if len(self._last_world) else 1.0
        radius = max(radius, 1e-3)
        eye = center + np.array([radius, -radius, radius * 0.6])
        for client in list(self.server.get_clients().values()):
            try:
                client.camera.position = eye
                client.camera.look_at = center
            except Exception:
                pass

    def add_frame(self, idx, image, depth, conf, c2w, intr, key=None):
        # Scene names are keyed by `key` (default: the frame index). Windowed
        # mode renders the same global index once per window occurrence, so it
        # passes a window-tagged key instead of overwriting the object.
        name = key if key is not None else f"{idx:05d}"
        fx, fy, cx, cy = (float(v) for v in intr)
        H, W = depth.shape
        if not np.isfinite(depth).all() or not np.isfinite(c2w).all() or fx <= 0.0 or fy <= 0.0:
            # A degenerate frame (usually the first symptom of a diverged
            # pose/depth output) renders as a frustum only, so one bad frame
            # cannot poison the whole live scene with NaN points.
            self.degenerate += 1
            if self.degenerate == 1 or self.degenerate % 10 == 0:
                print(f"warning: frame {idx} has non-finite depth/pose or zero "
                      f"intrinsics (degenerate frames so far: {self.degenerate})")
            self.count += 1
            return
        step = max(1, self.stride)
        keep = conf[::step, ::step] > self.conf_thr
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        d = depth[::step, ::step]
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cam = (xs - cx) * d / fx
            y_cam = (ys - cy) * d / fy
            pts_cam = np.stack([x_cam, y_cam, d], axis=-1).reshape(-1, 3)
            keep = keep.reshape(-1)
            R, t = c2w[:3, :3], c2w[:3, 3]
            world = pts_cam @ R.T + t
            world = world[keep]
        colors = (image.transpose(1, 2, 0)[::step, ::step].reshape(-1, 3)[keep] * 255).clip(0, 255).astype(np.uint8)
        if world.size == 0 or not np.isfinite(world).all():
            world = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)
        self.server.scene.add_point_cloud(
            f"/frames/{name}", points=world, colors=colors,
            point_size=self.point_size, point_shape="circle")
        # camera frustum (viser cameras are OpenCV-convention, like the model's c2w)
        scale = None
        pos = t
        if self._last_world is not None and self._last_world.size:
            cam_pts = [m[:4, 3] for m in getattr(self, "_cam_mats", [c2w])]
        if not hasattr(self, "_cam_mats"):
            self._cam_mats = []
        self._cam_mats.append(c2w)
        if len(self._cam_mats) >= 2:
            dists = [np.linalg.norm(self._cam_mats[i][:4, 3] - self._cam_mats[i - 1][:4, 3])
                     for i in range(1, len(self._cam_mats))]
            med = float(np.median(dists))
            if med > 0:
                scale = med * 0.35
        try:
            self.server.scene.add_camera_frustum(
                f"/cams/{name}",
                fov=2.0 * math.atan2(H / 2.0, fy),
                aspect=W / H,
                wxyz=self._quat_wxyz(R),
                position=pos,
                scale=scale if scale else 0.05,
                color=(60, 145, 255),
            )
        except Exception:
            pass
        self._last_world = world
        self.count += 1
        elapsed = time.time() - self.t0
        eta = elapsed / self.count * (self.total - self.count) if self.count else 0.0
        self._counter.value = f"{self.count} / {self.total}"
        self._elapsed.value = f"{elapsed:.1f}s"
        self._eta.value = f"{eta / 60.0:.1f} min" if eta >= 90.0 else f"{eta:.0f}s"
        self._status.value = "streaming frames..."
        if self.count in (1, 10, 50, 120):
            self._fit_view()

    def finish(self):
        self._status.value = "done — switching to the full viewer..."
        try:
            self.server.stop()
        except Exception:
            pass


def run_ggml_inference(args, images, height, width, on_frame=None):
    """Run the native CLI on [S,3,H,W] float32 frames; return raw outputs.

    With `on_frame`, the CLI streams every completed frame into a temp
    directory (`FRAME i/n` lines on stdout); each line triggers `on_frame`
    with that frame's depth/conf/c2w/intrinsics so the caller can render it
    live. stdout is relayed line-by-line instead of being captured, so the
    console shows progress while the 286-frame stream runs.
    """
    cli = args.build / "lingbot-map-cli"
    if not cli.is_file():
        raise SystemExit(f"GGML CLI not found at {cli}; run run_gui.sh to build it")
    model = args.model
    if not Path(model).is_file():
        raise SystemExit(
            f"GGUF model not found: {model}\n"
            "Download it per cpp_ggml/models/MODEL_CARD.md, e.g.\n"
            "  curl -L -o cpp_ggml/models/gguf/lingbot-map-q8.gguf "
            "https://huggingface.co/Asher-1/lingbot-map-gguf/resolve/main/lingbot-map-q8.gguf\n"
            "  (long-checkpoint variants: lingbot-map-long-{f32,f16,q8}.gguf in the same repo)")
    with tempfile.TemporaryDirectory(prefix="ggml-demo-") as tmp:
        bin_path = Path(tmp) / "frames.f32"
        out_path = Path(tmp) / "out.lbo"
        stream_dir = None
        cmd = [str(cli), str(model), str(bin_path), args.backend,
               str(height), str(width), str(out_path), str(len(images))]
        # the stream dir is needed for the live callback AND for native
        # skyseg (the CLI writes masked streamed frames through it)
        native_skyseg = args.mask_sky and Path(args.skyseg).is_file()
        if on_frame is not None or native_skyseg:
            stream_dir = Path(tmp) / "stream"
            stream_dir.mkdir()
            cmd.append(str(stream_dir))
        if native_skyseg:
            cmd.append(args.skyseg)   # argv[9]: native skyseg (CLI zeroes conf)
            # argv[10]/argv[11]: official-style mask cache + visualization
            # dirs (the CLI writes/reads the PNGs natively, same format as
            # <folder>_sky_masks)
            cmd.append(args.sky_mask_dir or "")
            cmd.append(args.sky_mask_visualization_dir or "")
        images.tofile(bin_path)
        # Explicit options: the CLI's named flags replaced the former
        # LINGBOT_* environment switches (see tools/lingbot-map-cli.cpp).
        # --kv_f16 wins; backward-compatible fallback to LINGBOT_KV_CACHE_F16:
        #   "1" (default) -> strict, "flash" -> flash, "0" -> none.
        if args.kv_f16 is not None:
            _f16_flag = args.kv_f16
        else:
            _f16_raw = os.environ.get("LINGBOT_KV_CACHE_F16", "1")
            _f16_flag = {"0": "none", "1": "strict", "flash": "flash"}.get(
                _f16_raw.lower(), "strict")
        cmd += [
            "--kv-f16", _f16_flag,
            "--kv-scale", str(args.kv_cache_scale),
            "--kv-window", str(args.kv_cache_window),
            "--scale-frames", str(args.num_scale_frames),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, bufsize=1)
        err_tail = []
        frame_done = 0
        err_thread = threading.Thread(
            target=lambda: err_tail.extend(l.rstrip() for l in proc.stderr), daemon=True)
        err_thread.start()
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("FRAME ") and on_frame is not None and stream_dir is not None:
                try:
                    done_text = line.split()[1]
                    idx = int(done_text.split("/")[0]) - 1
                except (IndexError, ValueError):
                    continue
                frame = None
                for _ in range(20):  # the writer flushes before printing, but be safe
                    frame = read_stream_frame(stream_dir / f"frame_{idx:04d}.bin")
                    if frame is not None:
                        break
                    time.sleep(0.05)
                if frame is not None:
                    frame_done += 1
                    on_frame(frame["index"], frame)
            elif line.startswith("RESULT"):
                print(line, flush=True)
            elif line:
                print(line, flush=True)
        proc.wait()
        err_thread.join(timeout=2.0)
        if proc.returncode != 0:
            # Show the lines that actually describe the failure: skip the
            # per-frame "DBG infer" chatter and keep the real error plus a
            # little context around it.
            err_lines = [l for l in (err_tail or []) if not l.startswith("DBG infer")]
            tail = "\n".join(err_lines[-25:] if err_lines
                             else [f"(no non-DBG stderr output; exit code {proc.returncode})"])
            hint = ""
            if "allocation" in tail.lower() or "out of device memory" in tail.lower() or "out of memory" in tail.lower():
                hint = (
                    "\nhint: the 8-frame scale-pass graph dominates this allocation.\n"
                    "  - prefer the CUDA backend (smaller reservations):"
                    " --backend CUDA0 --build cpp_ggml/build-cuda\n"
                    "  - Vulkan needs ~22.4 GB at 518x378 and ~28.5 GB at 518x518"
                    " for scale=8/window=64; ~13 GB at 448x336\n"
                    "  - or fall back to the light release profile"
                    " (--kv_cache_scale 1 --kv_cache_window 4 --num_scale_frames 1),"
                    " which streams ~5x faster in ~8 GB")
            raise SystemExit(f"GGML inference failed (exit code {proc.returncode}, "
                             f"after {frame_done} completed frames):\n{tail}{hint}")
        for line in (err_tail or []):
            if line.startswith("RESULT"):
                print(line, flush=True)
        pose_enc, depth = read_lbo(out_path, len(images), height, width)
        c2w, intr, conf = read_postprocess(Path(str(out_path) + ".post"), len(images))
    return pose_enc, depth, c2w, intr, conf


# ---------------------------------------------------------------------------
# Windowed inference (demo.py --mode windowed / GCTStream.inference_windowed
# alignment). Each window is the exact streaming primitive above run with a
# fresh KV cache (a fresh CLI process == inference_windowed's per-window
# clean_kv_cache); consecutive windows are then brought into the first
# window's coordinate frame with a pairwise similarity (s, R, t) estimated on
# the overlap and concatenated with a de-duplicating slice table. The
# alignment math below is a numpy port of gct_stream_window.py's
# _pairwise_alignment / _warp_predictions / _stitch_windows at the demo.py
# windowed defaults (keyframe_interval=1: every frame is a keyframe, so the
# anchor pair is the last overlap frame and the depth-ratio scale aggregates
# the whole overlap).
# ---------------------------------------------------------------------------

def _quat_to_mat(q):
    """Scalar-last (x, y, z, w) quaternions [..., 4] -> matrices [..., 3, 3].

    Port of lingbot_map.utils.rotation.quat_to_mat (PyTorch3D).
    """
    i, j, k, r = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / np.maximum((q * q).sum(axis=-1), 1e-30)
    o = np.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), axis=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def _mat_to_quat(m):
    """Rotation matrices [..., 3, 3] -> scalar-last quaternions (rotation.py)."""
    m = m.reshape(m.shape[:-2] + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = (m[..., i] for i in range(9))
    q_abs = np.sqrt(np.maximum(np.stack((
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22), axis=-1), 0.0))
    quat_by_rijk = np.stack((
        np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
        np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
        np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
        np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
    ), axis=-2)
    quat_candidates = quat_by_rijk / (2.0 * np.maximum(q_abs[..., None], 0.1))
    idx = q_abs.argmax(axis=-1)  # best-conditioned candidate (largest denominator)
    out = np.take_along_axis(quat_candidates, idx[..., None, None], axis=-2)[..., 0, :]
    out = out[..., [1, 2, 3, 0]]  # rijk -> ijkr
    return np.where(out[..., 3:4] < 0, -out, out)  # standardize: w >= 0


def _split_windows(total, eff_window, eff_overlap):
    """Window list per inference_windowed's fixed-interval branch."""
    if eff_window >= total:
        return [(0, total)]
    windows = []
    step = max(eff_window - eff_overlap, 1)
    for start in range(0, total, step):
        end = min(start + eff_window, total)
        if end - start >= eff_overlap or end == total:
            windows.append((start, end))
        if end == total:
            break
    return windows


def _pairwise_alignment(prev, curr, overlap):
    """(s, R, t) mapping curr into prev's frame, estimated on the overlap.

    keyframe_interval=1: the anchor is the last overlap frame and the
    depth-ratio scale aggregates every overlap frame (the paired-keyframe
    mask of _pairwise_alignment is all-true).
    """
    s = np.float32(1.0)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    if overlap <= 0:
        return s, R, t
    total_prev = prev["pose_enc"].shape[0]
    total_curr = curr["pose_enc"].shape[0]
    start = max(total_prev - overlap, 0)
    eff = min(overlap, total_prev - start, total_curr)
    if eff <= 0:
        return s, R, t
    da = prev["depth"][start:start + eff]   # (eff, H, W)
    db = curr["depth"][:eff]
    ok = np.isfinite(da) & np.isfinite(db) & (np.abs(db) > np.finfo(np.float32).eps)
    if ok.any():
        s = np.float32(np.clip(np.median(da[ok] / db[ok]), 1e-3, 1e3))
    idx_a = start + eff - 1
    Ra = _quat_to_mat(prev["pose_enc"][idx_a, 3:7].astype(np.float32))
    Rb = _quat_to_mat(curr["pose_enc"][eff - 1, 3:7].astype(np.float32))
    ca = prev["pose_enc"][idx_a, :3].astype(np.float32)
    cb = curr["pose_enc"][eff - 1, :3].astype(np.float32)
    R_ab = Ra @ Rb.T                          # Ra = R_ab @ Rb
    t_ab = ca - s * (R_ab @ cb)               # ca = s * R_ab @ cb + t_ab
    return s, R_ab.astype(np.float32), t_ab.astype(np.float32)


def _warp_window(win, s, R, t):
    """Apply the official window warp: pose_enc (center+quat) and c2w.

    Matches _warp_predictions exactly. pose_enc encodes the w2c pose — the
    decoder emits w2c = [quat_mat | pe[:3]] (verified against
    pose_encoding_to_extri_intri, 0.0 deviation), so the decoder image of the
    warped pose_enc for the c2w sidecar is R'_c2w = R_c2w @ R.T and
    t'_c2w = s * t_c2w - R_c2w @ (R.T @ t); warping the sidecar this way keeps
    the merged c2w equal to decode(merged pose_enc) with no torch decoder.
    """
    pe = win["pose_enc"].copy()
    rot = _quat_to_mat(pe[:, 3:7])
    pe[:, :3] = s * np.einsum("ab,nb->na", R, pe[:, :3]) + t
    pe[:, 3:7] = _mat_to_quat(np.einsum("ab,nbc->nac", R, rot))
    out = dict(win)
    out["pose_enc"] = pe
    out["depth"] = win["depth"] * s
    Rt = R.T
    c2w = win["c2w"].copy()
    c2w[:, :3, :3] = np.einsum("nij,jk->nik", win["c2w"][:, :3, :3], Rt)
    c2w[:, :3, 3] = (s * win["c2w"][:, :3, 3]
                     - np.einsum("nij,j->ni", win["c2w"][:, :3, :3], Rt @ t))
    out["c2w"] = c2w
    return out


def _stitch_windows(windows, overlap):
    """Concatenate per-window predictions, de-duplicating the overlaps.

    Every non-final window contributes [0, total - overlap) frames — the
    slice table of gct_stream_window.py:_stitch_windows.
    """
    n = len(windows)
    out = {}
    for k in windows[0]:
        parts = []
        for i, w in enumerate(windows):
            end = w[k].shape[0] if i == n - 1 else max(w[k].shape[0] - overlap, 0)
            if end > 0:
                parts.append(w[k][:end])
        out[k] = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return out


def run_ggml_windowed(args, images, height, width, on_frame=None):
    """Windowed inference over [S,3,H,W], aligned with demo.py --mode windowed.

    Windows follow inference_windowed's fixed-interval rules with the demo.py
    windowed defaults (keyframe_interval=1): actual window frames =
    window_size, overlap from --overlap_keyframes (clamped up to
    num_scale_frames) else --overlap_size. Every window runs the streaming
    primitive in a fresh CLI process (fresh KV cache), then windows are
    similarity-aligned into the first window's frame and stitched.
    """
    S = images.shape[0]
    ws = max(1, min(args.num_scale_frames, S))
    kf_int = 1  # the native engine streams every frame (no skip_append)
    if args.overlap_keyframes is not None:
        eff_overlap = max(ws, args.overlap_keyframes * kf_int)
    elif args.overlap_size is not None:
        eff_overlap = args.overlap_size
    else:
        eff_overlap = ws
    eff_overlap = min(eff_overlap, S - 1) if S > 1 else 0
    eff_window = min(ws + max(args.window_size - ws, 0) * kf_int, S)
    windows = _split_windows(S, eff_window, eff_overlap)
    print(f"Windowed inference: {len(windows)} window(s) {windows} "
          f"(window_size={args.window_size}, overlap={eff_overlap}, scale={ws})")

    # Native skyseg caches masks by CLI-run-local frame index; give each
    # window its own cache/visualization subdirectory so a windowed run
    # never reuses another window's mask for a different physical frame.
    win_args = args
    if args.mask_sky and Path(args.skyseg).is_file():
        import copy
        win_args = copy.copy(args)
        base_m = args.sky_mask_dir or "ggml_sky_masks"
        base_v = args.sky_mask_visualization_dir or (base_m + "_vis")

    warped = []
    for wi, (start, end) in enumerate(windows):
        if win_args is not args:
            win_args.sky_mask_dir = f"{base_m}/win{wi:02d}"
            win_args.sky_mask_visualization_dir = f"{base_v}/win{wi:02d}"
        cb = None
        if on_frame is not None:
            cb = (lambda idx, f, _s=start, _w=wi: on_frame(_s + idx, f, _w))
        print(f"[window {wi + 1}/{len(windows)}] frames [{start}, {end}) ...")
        pose_enc, depth, c2w, intr, conf = run_ggml_inference(
            win_args, images[start:end], height, width, on_frame=cb)
        win = {"pose_enc": pose_enc, "depth": depth, "depth_conf": conf,
               "c2w": c2w, "intrinsics": intr}
        if wi > 0:
            s_ab, R_ab, t_ab = _pairwise_alignment(warped[-1], win, eff_overlap)
            print(f"[window {wi + 1}] alignment scale={s_ab:.6f} "
                  f"t=[{t_ab[0]:.4f} {t_ab[1]:.4f} {t_ab[2]:.4f}]")
            win = _warp_window(win, s_ab, R_ab, t_ab)
        warped.append(win)
    merged = _stitch_windows(warped, eff_overlap)
    return (merged["pose_enc"], merged["depth"], merged["c2w"],
            merged["intrinsics"], merged["depth_conf"])


def build_pred_dict(args, images, depth, c2w, intr, conf):
    """Assemble the exact pred_dict contract of PointCloudViewer._process_pred_dict.

    The official demo stores camera-to-world under "extrinsic"; the GGML CLI
    emits the same convention, so no inversion is needed.
    """
    S, _, H, W = images.shape
    intrinsic = np.zeros((S, 3, 3), dtype=np.float32)
    intrinsic[:, 0, 0] = intr[:, 0]
    intrinsic[:, 1, 1] = intr[:, 1]
    intrinsic[:, 0, 2] = intr[:, 2]
    intrinsic[:, 1, 2] = intr[:, 3]
    intrinsic[:, 2, 2] = 1.0
    return {
        "images": images,                       # (S, 3, H, W) float [0, 1]
        "depth": depth[:, :, :, None],          # (S, H, W, 1)
        "depth_conf": conf.reshape(S, H, W),    # (S, H, W)
        "extrinsic": c2w[:, :3, :4],            # (S, 3, 4) camera-to-world
        "intrinsic": intrinsic,                 # (S, 3, 3)
    }


def main():
    ap = argparse.ArgumentParser(description="GGML streaming 3D reconstruction GUI (official viewer)")
    ap.add_argument("--image_folder", type=str, default="example/courthouse")
    ap.add_argument("--image_ext", type=str, default=".jpg,.png,.JPG",
                    help="comma-separated extensions, as in demo.py")
    ap.add_argument("--model", type=str, default="cpp_ggml/models/gguf/lingbot-map-q8.gguf")
    ap.add_argument("--build", type=Path, default=Path("cpp_ggml/build-vulkan"))
    ap.add_argument("--backend", type=str, default="Vulkan0")
    ap.add_argument("--frames", type=int, default=0, help="0 = all frames in the folder")
    # Preprocessing, aligned with demo.py --image_size / --patch_size.
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int,
                    help="override the official crop rule; distorts the aspect ratio")
    ap.add_argument("--width", type=int,
                    help="override the target width (default --image_size)")
    # Streaming cache profile, aligned with demo.py (num_scale_frames=8,
    # kv_cache_sliding_window=64).
    ap.add_argument("--kv_cache_scale", type=int, default=8)
    ap.add_argument("--kv_cache_window", type=int, default=64)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    # Inference mode + windowed options, aligned with demo.py --mode
    # streaming|windowed (--window_size / --overlap_size / --overlap_keyframes).
    ap.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
                    help="streaming: frame-by-frame with the persistent KV cache; "
                         "windowed: overlapping windows for long sequences — each "
                         "window is a fresh-cache streaming run, then windows are "
                         "similarity-aligned and stitched (inference_windowed)")
    ap.add_argument("--window_size", type=int, default=64,
                    help="frames per window (windowed mode)")
    ap.add_argument("--overlap_size", type=int, default=16,
                    help="overlap between windows in frames (windowed mode)")
    ap.add_argument("--overlap_keyframes", type=int, default=None,
                    help="overlap in keyframes; overrides --overlap_size and is "
                         "clamped up to --num_scale_frames (windowed mode)")
    # Viewer parameters, aligned with demo.py
    ap.add_argument("--kv_f16", type=str, default=None,
                    choices=["strict", "flash", "none"],
                    help="KV cache precision contract (default: strict, or the "
                         "LINGBOT_KV_CACHE_F16 env). 'none' = exact F32 cache: "
                         "tightest parity (2000-frame long stream: pose 8.6e-05 "
                         "vs strict 5.9e-04) and no F16-cache depth scatter tail "
                         "(0.00%%), at roughly +4 GB resident payload and ~2x wall")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--conf_threshold", type=float, default=1.5)
    ap.add_argument("--downsample_factor", type=int, default=10)
    ap.add_argument("--point_size", type=float, default=0.00001)
    ap.add_argument("--mask_sky", action="store_true",
                    help="zero the depth confidence of sky pixels; with the ggml "
                         "engine this runs the NATIVE skyseg GGUF inside the CLI "
                         "(no onnxruntime needed) — see cpp_ggml/scripts/convert_skyseg.py")
    ap.add_argument("--skyseg", type=str, default="cpp_ggml/models/gguf/lingbot-map-skyseg-f16.gguf",
                    help="skyseg GGUF used when --mask_sky is set (ggml engine)")
    ap.add_argument("--sky_model", type=str, default="skyseg.onnx")
    ap.add_argument("--depth_stride", type=int, default=1)
    ap.add_argument("--sky_mask_dir", type=str, default=None,
                    help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    ap.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                    help="Save sky mask visualizations (original | mask | overlay) here")
    # Input alternatives aligned with demo.py: video input, frame decimation,
    # rotation, and preprocessed-image export.
    ap.add_argument("--video_path", type=str, default=None,
                    help="mp4/etc video instead of --image_folder; sampled at --fps")
    ap.add_argument("--fps", type=int, default=10, help="sampling fps for --video_path")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every N-th image of the folder (demo.py semantics)")
    ap.add_argument("--rotate_clockwise_90", action="store_true",
                    help="rotate source images 90° CW before preprocessing")
    ap.add_argument("--export_preprocessed", type=str, default=None,
                    help="export stride-sampled, resized/cropped images to this folder")
    # Live streaming view during the native inference run.
    ap.add_argument("--streaming_view", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="grow a live viser scene (point clouds + frustums) while the "
                         "native runtime streams; disable with --no-streaming_view")
    ap.add_argument("--stream_stride", type=int, default=4,
                    help="point-cloud decimation for the live streaming view")
    args = ap.parse_args()

    folder = Path(args.image_folder)
    assert args.image_folder or args.video_path, \
        "provide --image_folder or --video_path (as in demo.py)"

    resolved_folder = str(folder) if args.image_folder else None
    if args.video_path:
        files, resolved_folder = extract_video_frames(args.video_path, args.fps)
    else:
        files = sorted(p for ext in args.image_ext.split(",") for p in folder.glob(f"*{ext}"))
    if args.frames > 0:
        files = files[:args.frames]
    if args.stride > 1:
        files = files[::args.stride]
    if not files:
        raise SystemExit(f"no {args.image_ext} frames in {folder}")

    if args.rotate_clockwise_90:
        import tempfile
        rot_dir = tempfile.mkdtemp(prefix="lingbot_ggml_rot_cw90_")
        rotated = []
        for p in files:
            out = os.path.join(rot_dir, os.path.basename(p))
            Image.open(p).transpose(Image.ROTATE_270).save(out)
            rotated.append(out)
        files = rotated
        resolved_folder = rot_dir
        print(f"Rotated {len(files)} images 90° clockwise → {rot_dir}")

    print(f"Loading {len(files)} frames from {resolved_folder}...")
    images = load_official_frames(files, args.image_size, args.patch_size,
                                  args.height, args.width)
    height, width = images.shape[2], images.shape[3]
    print(f"Preprocessed to {width}x{height} (WxH); "
          f"scale={args.kv_cache_scale} window={args.kv_cache_window}")

    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        import cv2
        for i in range(images.shape[0]):
            img = (images[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Exported {images.shape[0]} preprocessed images to {args.export_preprocessed}")

    stream = None
    if args.streaming_view:
        try:
            stream = StreamingViewer(
                port=args.port, total_frames=len(files),
                image_h=height, image_w=width,
                conf_threshold=args.conf_threshold,
                stride=args.stream_stride,
                point_size=args.point_size)
            print(f"Live streaming view: http://localhost:{args.port} "
                  f"(point clouds and frustums appear as frames complete)")
        except Exception as exc:
            print(f"streaming view unavailable ({exc}); falling back to console progress")
            stream = None

    print(f"Running native GGML inference ({args.mode}, streams frame-by-frame)...")
    try:
        def on_frame(idx, frame, tag=0):
            if stream is not None:
                # Windowed mode renders the same global index once per window
                # occurrence; tag the scene names so nothing is overwritten.
                key = f"w{tag:02d}_{idx:05d}" if tag else f"{idx:05d}"
                stream.add_frame(idx, images[idx], frame["depth"],
                                 frame["depth_conf"], frame["c2w"],
                                 frame["intrinsics"], key=key)
        runner = run_ggml_windowed if args.mode == "windowed" else run_ggml_inference
        pose_enc, depth, c2w, intr, conf = runner(
            args, images, height, width, on_frame=on_frame if stream else None)
    except SystemExit:
        if stream is not None:
            stream.finish()
        raise
    print(f"Reconstructed {len(images)} frames; pose_enc[0]={np.round(pose_enc[0], 4)}")

    if stream is not None:
        stream.finish()
        time.sleep(0.5)  # let the OS release the port for the official viewer

    pred_dict = build_pred_dict(args, images, depth, c2w, intr, conf)

    try:
        from lingbot_map.vis import PointCloudViewer
    except ImportError:
        raise SystemExit("viser not installed. Install with: pip install lingbot-map[vis]")

    viewer = PointCloudViewer(
        pred_dict=pred_dict,
        port=args.port,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        mask_sky=False,  # sky masking is done natively in the CLI when
                         # --mask_sky was set (see above); the onnxruntime
                         # path is not needed for the ggml engine
        image_folder=resolved_folder,
        skyseg_model_path=args.sky_model,
        sky_mask_dir=args.sky_mask_dir,
        sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        depth_stride=args.depth_stride,
    )
    print(f"Opening the official reconstruction viewer at http://localhost:{args.port}")
    viewer.run()


if __name__ == "__main__":
    main()
