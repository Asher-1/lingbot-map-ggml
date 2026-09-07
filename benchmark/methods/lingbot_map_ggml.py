"""BSS adapter for the native LingBot-Map GGML streaming runtime.

The model graph and pose/FoV postprocess run in C++. Python here only bridges
the benchmark's BSS artifact format to the CLI's documented float32 input.
"""

import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from benchmark.core.loader import BSSLoader
from benchmark.method.base import BaseMethod


_LBO1 = 0x4C424F31
_LBP1 = 0x4C425031
_LBP2 = 0x4C425032


def _read_lbo(path: Path, frames: int, height: int, width: int):
    raw = path.read_bytes()
    magic, pose_count, depth_count = struct.unpack("<III", raw[:12])
    expected_depth = frames * height * width
    if magic != _LBO1 or pose_count != frames * 9 or depth_count != expected_depth:
        raise ValueError(f"invalid GGML LBO1 output: {path}")
    values = np.frombuffer(raw, dtype=np.float32, offset=12)
    if values.size != pose_count + depth_count:
        raise ValueError(f"truncated GGML LBO1 output: {path}")
    return values[pose_count:].reshape(frames, height, width).copy()


def _read_postprocess(path: Path, frames: int):
    raw = path.read_bytes()
    magic, c2w_count, intrinsics_count = struct.unpack("<III", raw[:12])
    if magic == _LBP1:
        values = np.frombuffer(raw, dtype=np.float32, offset=12)
        if c2w_count != frames * 16 or intrinsics_count != frames * 4 or values.size != c2w_count + intrinsics_count:
            raise ValueError(f"truncated GGML LBP1 postprocess output: {path}")
        return values[:c2w_count].reshape(frames, 4, 4).copy(), values[c2w_count:].reshape(frames, 4).copy(), None
    if magic != _LBP2 or len(raw) < 16:
        raise ValueError(f"invalid GGML LBP postprocess output: {path}")
    confidence_count, = struct.unpack("<I", raw[12:16])
    values = np.frombuffer(raw, dtype=np.float32, offset=16)
    if (c2w_count != frames * 16 or intrinsics_count != frames * 4 or
            values.size != c2w_count + intrinsics_count + confidence_count):
        raise ValueError(f"truncated GGML LBP2 postprocess output: {path}")
    start = c2w_count + intrinsics_count
    return (
        values[:c2w_count].reshape(frames, 4, 4).copy(),
        values[c2w_count:start].reshape(frames, 4).copy(),
        values[start:].copy(),
    )


class LingbotMapGgmlMethod(BaseMethod):
    """Run native C++ GGML inference while preserving the BSS output contract."""

    def __init__(
        self,
        model: str,
        build: str,
        backend: str = "CUDA0",
        kv_cache_scale_frames: int = 1,
        kv_cache_sliding_window: int = 4,
        area_budget: Optional[int] = 255000,
        align: int = 14,
        logger=None,
        **_: Any,
    ):
        super().__init__(area_budget=area_budget, align=align, logger=logger)
        repo_root = Path(__file__).resolve().parents[2]
        model_path = Path(model).expanduser()
        build_path = Path(build).expanduser()
        self.model = (model_path if model_path.is_absolute() else repo_root / model_path).resolve()
        self.cli = ((build_path if build_path.is_absolute() else repo_root / build_path).resolve()
                    / "lingbot-map-cli")
        self.backend = backend
        self.kv_cache_scale_frames = int(kv_cache_scale_frames)
        self.kv_cache_sliding_window = int(kv_cache_sliding_window)
        if not self.model.is_file():
            raise FileNotFoundError(f"GGUF model does not exist: {self.model}")
        if not self.cli.is_file():
            raise FileNotFoundError(f"GGML CLI does not exist: {self.cli}")

    def process_scene(self, gt_artifact) -> Dict[str, Any]:
        loader = BSSLoader(gt_artifact, resize_context=self.resize_context)
        rgb = loader.load_rgb_list()
        if not rgb:
            raise ValueError("BSS scene has no RGB frames")
        height, width = rgb[0].shape[:2]
        if height % self.align or width % self.align:
            raise ValueError(f"BSS resize did not preserve {self.align}-pixel alignment: {height}x{width}")
        if any(frame.shape != (height, width, 3) for frame in rgb):
            raise ValueError("BSS scene frames have inconsistent dimensions")

        frames = np.stack(rgb, axis=0).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        with tempfile.TemporaryDirectory(prefix="lingbot-map-ggml-") as temp:
            temp_path = Path(temp)
            input_path = temp_path / "frames.f32"
            output_path = temp_path / "output.lbo"
            frames.tofile(input_path)
            env = os.environ.copy()
            env["LINGBOT_KV_CACHE_SCALE"] = str(self.kv_cache_scale_frames)
            env["LINGBOT_KV_CACHE_WINDOW"] = str(self.kv_cache_sliding_window)
            subprocess.run(
                [str(self.cli), str(self.model), str(input_path), self.backend,
                 str(height), str(width), str(output_path), str(len(rgb))],
                check=True, env=env,
            )
            depth = _read_lbo(output_path, len(rgb), height, width)
            c2w, intrinsics, confidence = _read_postprocess(Path(str(output_path) + ".post"), len(rgb))

        frame = {
            "rgb": rgb,
            "depth": [depth[i] for i in range(len(rgb))],
            "pose": [c2w[i] for i in range(len(rgb))],
            "intrinsics": [intrinsics[i] for i in range(len(rgb))],
        }
        if confidence is not None:
            frame["confidence"] = list(confidence.reshape(len(rgb), height, width))

        return {
            "frame": frame,
            "global": {},
        }
